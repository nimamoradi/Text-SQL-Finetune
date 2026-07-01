import os
import jax
import optax
import orbax.checkpoint as ocp
import grain.python as grain

from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout

from flax import nnx
import qwix
from tunix.sft import metrics_logger

# Assuming LLMFactoryConfig and LLMModuleFactory are updated for NNX
from src.model_loading.llm_factory import LLMModuleFactory, LLMFactoryConfig

class TextToSQLGRPOTrainer:
    def __init__(
            self,
            learning_rate: float,
            batch_size: int,
            num_generations: int,
            beta: float,
            lora_rank: int,
            checkpoint_dir: str = "./checkpoints",
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_generations = num_generations
        self.beta = beta
        self.lora_rank = lora_rank
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)

        self.base_model = None
        self.actor_model = None
        self.data_loader = None
        self.reward_functions = []
        self.modules = None

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Updated to use StandardCheckpointer for NNX state compatibility
        self.ckpt_manager = ocp.CheckpointManager(
            os.path.abspath(self.checkpoint_dir),
            ocp.StandardCheckpointer(),
            options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
        )

    def initialize_model(self, config: LLMFactoryConfig, load_step: int = None):
        """Build NNX base model, apply LoRA, and load checkpoint if present."""
        self.config = config

        # 1. Build the NNX base model via the factory
        self.modules = LLMModuleFactory.build(config)
        self.base_model = self.modules.model  # This is the stateful NNX model

        # 2. Apply LoRA to create the Actor model
        print("Applying LoRA to the NNX model...")
        self._setup_lora_actor()

        # 3. Restore LoRA weights if a checkpoint exists
        self.load_checkpoint(step=load_step)

    def _setup_lora_actor(self):
        """Wraps the base NNX model with LoRA adapters."""
        lora_provider = qwix.LoraProvider(
            module_path=(
                ".*q_einsum|.*kv_einsum"
            ),
            rank=self.lora_rank,
            alpha=float(self.lora_rank),
        )

        model_input = self.base_model.get_model_input()
        self.actor_model = qwix.apply_lora_to_model(
            self.base_model, lora_provider, **model_input
        )

        # Ensure the Actor model's state respects the sharding constraints
        with self.config.mesh:
            state = nnx.state(self.actor_model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(self.actor_model, sharded_state)


    def add_reward_function(self, func):
        self.reward_functions.append(func)

    def _get_eos_tokens(self):
        if self.modules is None:
            return [1]
        tok = getattr(self.modules, "tokenizer", None)
        if tok is None:
            return [1]
        
        eos_tokens = []
        if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
            eos_tokens.append(tok.eos_token_id)
        
        eos_method = getattr(tok, "eos_id", None)
        if callable(eos_method):
            eos_val = eos_method()
            if eos_val not in eos_tokens:
                eos_tokens.append(eos_val)
                
        if not eos_tokens:
            eos_tokens = [1]
            
        # Dynamically append '<end_of_turn>' token ID for Gemma models
        if hasattr(tok, "piece_to_id"):
            try:
                eot_id = tok.piece_to_id("<end_of_turn>")
                if eot_id > 3 and eot_id not in eos_tokens:
                    eos_tokens.append(eot_id)
            except Exception:
                pass
        elif hasattr(tok, "convert_tokens_to_ids"):
            try:
                eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
                if eot_id is not None and eot_id not in eos_tokens:
                    eos_tokens.append(eot_id)
            except Exception:
                pass
                
        return eos_tokens

    def train(self, steps: int):
        if (
                self.base_model is None
                or self.actor_model is None
                or self.data_loader is None
                or not self.reward_functions
        ):
            raise ValueError(
                "Pipeline components missing. Initialize model, data, and rewards first."
            )

        print(f"Starting GRPO training for {steps} steps...")

        # 1. Setup Optimizer Strategy
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.learning_rate,
            warmup_steps=min(100, steps // 10),
            decay_steps=steps,
            end_value=0.0,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=schedule, weight_decay=0.01),
        )

        # 2. Extract Mesh from Config (No need to recreate it)
        mesh = self.config.mesh

        # Configure Metrics Logger
        log_dir = os.path.join(self.checkpoint_dir, "tensorboard_logs")
        metrics_logging_options = metrics_logger.MetricsLoggerOptions(
            log_dir=log_dir,
            flush_every_n_steps=10
        )

        # 3. Formulate Training and Generation Rule Bounds
        training_config = rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            eval_every_n_steps=max(1, steps // 5),
            max_steps=steps,
            mini_batch_size=self.batch_size,
            train_micro_batch_size=self.batch_size,
            checkpoint_root_directory=self.checkpoint_dir,
            metrics_logging_options=metrics_logging_options,
        )

        eos_tokens = self._get_eos_tokens()

        rollout_config = base_rollout.RolloutConfig(
            max_tokens_to_generate=256,
            max_prompt_length=1060,
            kv_cache_size=1060 + 256,
            temperature=0.9,
            top_p=0.9,
            top_k=40,
            eos_tokens=eos_tokens,
        )

        # 4. Construct Cluster Mesh Roles Configuration
        cluster_config = rl_cluster_lib.ClusterConfig(
            role_to_mesh={
                rl_cluster_lib.Role.ACTOR: mesh,
                rl_cluster_lib.Role.REFERENCE: mesh,
                rl_cluster_lib.Role.ROLLOUT: mesh,
            },
            rollout_engine="vllm",
            offload_to_cpu=False,
            training_config=training_config,
            rollout_config=rollout_config,
        )

        # 5. Map Hyperparameters
        algo_config = GRPOConfig(
            num_generations=self.num_generations,
            num_iterations=1,
            beta=self.beta,
            epsilon=0.0,
        )

        # 6. Instantiate Node Topology Cluster
        cluster = rl_cluster_lib.RLCluster(
            actor=self.actor_model,  # LoRA Policy Model
            reference=self.base_model,  # Frozen Base Model
            tokenizer=self.modules.tokenizer,
            cluster_config=cluster_config,
        )

        # 7. Hand over to Learner Core
        learner = GRPOLearner(
            rl_cluster=cluster,
            algo_config=algo_config,
            reward_fns=self.reward_functions,
            data_shuffle_seed=42,
        )

        # limited_dataloader = itertools.islice(self.data_loader, steps)

        print("Handing control to Tunix internal training loop...")
        learner.train(
            train_ds=self.data_loader,
            eval_ds=None,
            skip_jit=False,
        )

        self.save_checkpoint(steps)
        print("Training successfully completed!")

    def save_checkpoint(self, step: int):
        if self.actor_model is None:
            return

        print(f"Saving LoRA checkpoint at step {step}...")
        # Extract ONLY the LoRA parameters to save space
        lora_state = nnx.state(self.actor_model, nnx.LoRAParam)

        self.ckpt_manager.save(
            step,
            args=ocp.args.StandardSave(lora_state),
        )
        self.ckpt_manager.wait_until_finished()

    def load_checkpoint(self, step: int = None):
        """Restore only the LoRA params into the Actor model if a checkpoint exists."""
        if step is not None and step not in self.ckpt_manager.all_steps():
            raise ValueError(f"Requested checkpoint step {step} not found. Available steps: {list(self.ckpt_manager.all_steps())}")
            
        target_step = step if step is not None else self.ckpt_manager.latest_step()

        if target_step is None:
            print("No checkpoints found. Using base weights.")
            return

        print(f"Restoring LoRA checkpoint from step {target_step}...")

        # 1. Define the structural target for Orbax
        target_state = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
            nnx.state(self.actor_model, nnx.LoRAParam)
        )

        # 2. Restore the state from disk
        restored_lora_state = self.ckpt_manager.restore(
            target_step,
            args=ocp.args.StandardRestore(item=target_state),
        )

        # 3. Update the running model with the restored LoRA state
        nnx.update(
            self.actor_model,
            jax.tree.map(
                lambda a, b: b,
                nnx.state(self.actor_model, nnx.LoRAParam),
                restored_lora_state
            ),
        )
        print("Checkpoint restored successfully.")
local train_size = 10000;
local batch_size = 8;
local gradient_accumulation_batch_size = 2;
local num_epochs = 4;
local learning_rate = 5e-7;
local weight_decay = 0.1;
local warmup_ratio = 0.06;
local transformer_model = "roberta-large";
local cuda_device = 0;

{
  "dataset_reader": {
    "type": "transformer_binary_qa",
    "sample": -1,
    "pretrained_model": transformer_model,
    "max_pieces": 256
  },
  "validation_dataset_reader": {
    "type": "transformer_binary_qa",
    "sample": -1,
    "pretrained_model": transformer_model,
    "max_pieces": 256
  },
  //"datasets_for_vocab_creation": [],
  "train_data_path": "",
  "validation_data_path": "",

  "model": {
    "type": "transformer_binary_qa",
    "pretrained_model": transformer_model
  },
  "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size": 4
      }
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "weight_decay": 0.1,
      "betas": [0.9, 0.98],
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
      "lr": 1e-05
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": 4,
      "cut_frac": 0.1,
      "num_steps_per_epoch": 729,
    },
    "validation_metric": "+EM",
    "num_serialized_models_to_keep": 1,
    //"should_log_learning_rate": true,
    "num_gradient_accumulation_steps": 12,
    // "grad_clipping": 1.0,
    "num_epochs": 4,
    "cuda_device": cuda_device,
    "random_seed": 2
  }
}
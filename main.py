import training.omniglot_training

# omniglot_path = "tmp/omniglot"
# training.omniglot_training.train(omniglot_path)

omniglot_path = "/home/zelgunn/datasets/omniglot"
training.omniglot_training.train(omniglot_path, train_name="train", test_name="test")

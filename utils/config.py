class Config_Nodule:
    data_path = "./data/"
    save_path = "./result/"
    tensorboard_path = "./tensorboard/"
    load_path = save_path + "/checkpoint/xxx.pth"

    epochs = 200                 # number of total epochs to run (default: 400)
    batch_size = 32              # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 5               # the number of classes (background + foreground)
    img_size = (48, 48, 48)   # the input size of model,  default (100, 100, 100) (64, 64, 64) (48, 48, 48) (36, 36, 36)
    train_split = "train"        # the file name of training set
    val_split = "val"           # the file name of testing set
    test_split = "test"
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "class"     
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SimGro"
    num_templates = 3
    feature_dim = 128


def get_config(task="Nodule"):
    if task == "Nodule":
        return Config_Nodule()
    else:
        assert("We do not have the related dataset, please choose another task.")                    

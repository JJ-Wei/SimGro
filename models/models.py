from models.SimGro.pulmotarynetwork import SimGro

def get_model(modelname="SimGro", img_size=(48, 48, 48), img_channel=1, classes=9,num_templates=3, feature_dim=128):
    if modelname == "SimGro":
        model = SimGro(
            input_channels=img_channel, 
            num_nodule_classes=classes, 
            img_size=img_size,
            num_templates=num_templates, 
            feature_dim=feature_dim
        )
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model
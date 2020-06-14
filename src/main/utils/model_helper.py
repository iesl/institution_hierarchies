from main.models.SetTransformer import SetTransformer
from main.models.SetAvgEmb import SetAvgEmb
from main.models.OrderedLSTM import OrderedLSTM
from main.models.OrderedTransformer import OrderedTransformer

from main.baselines.TokenSim import TokenSim



def get_model(config, data_extractor, tokenizer):
    '''
    Returns an object of the model 
    
    param config: configuration of the model 
    param data_extractor:
    param tokenizer:
    return: model 
    '''
    model = None

    # Set up Model
    if config.model_name == "SetAvgEmb":
        model = SetAvgEmb(config, data_extractor, tokenizer)
    elif config.model_name == "SetTransformer":
        model = SetTransformer(config, data_extractor, tokenizer)
    elif config.model_name == "OrderedLSTM":
        model = OrderedLSTM(config, data_extractor, tokenizer)
    elif config.model_name == "OrderedTransformer":
        model = OrderedTransformer(config, data_extractor, tokenizer)
    else:
        raise ValueError("Model Unknown: ", config.model_name)

    return model


def get_baseline(config, data_extractor, tokenizer):
    '''
    Returns an object of the model

    param config: configuration of the model
    param data_extractor:
    param tokenizer:
    return: model
    '''
    model = None

    # Set up Model
    if config.model_name == "TokenSim":
        model = TokenSim(config, data_extractor, tokenizer)

    return model

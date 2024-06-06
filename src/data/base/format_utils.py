import torch

def video_tuple_to_dict(data):
    """
    the input data is a tuple of (visual, input_ids, attention_mask, labels)
    visual is 6D input: (batch_size, Image in same chunk, Time, num_channels, height, width)
    """
    visual, input_ids, attention_mask, labels = data[0], data[1][0], data[1][1], data[1][2]
    # print("visual shape: ", visual.shape)
    return {"visual": visual, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def raw_video_tuple_to_dict(data):
    """
    the input data is a tuple of (visual, input_ids, attention_mask, labels)
    visual is 6D input: (batch_size, Image in same chunk, Time, num_channels, height, width)
    """
    visual, input_ids, attention_mask, labels = data[0], data[1][0].squeeze(0), data[1][1].squeeze(0), data[1][2].squeeze(0)
    return {"visual": visual, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def tuple_to_dict(data):
    """
    the input data is a tuple of (visual, input_ids, attention_mask, labels)
    visual is 6D input: (batch_size, Image in same chunk, Time, num_channels, height, width)
    """
    visual, input_ids, attention_mask, labels = data[0].unsqueeze(1).unsqueeze(1), data[1][0], data[1][1], data[1][2]
    rendered_text_image = data[2].unsqueeze(1).unsqueeze(1)
    return {"visual": visual, "rendered_text_image": rendered_text_image, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def tuple_to_dict_w_text_image(data):
    """
    the input data is a tuple of (visual, input_ids, attention_mask, labels)
    visual is 6D input: (batch_size, Image in same chunk, Time, num_channels, height, width)
    """
    visual, input_ids, attention_mask, labels = data[0].unsqueeze(1).unsqueeze(1), data[1][0], data[1][1], data[1][2]
    rendered_text_image, rendered_text_token_ids = data[2][0].unsqueeze(1).unsqueeze(1), data[2][1]
    return {"visual": visual, "rendered_text_image": rendered_text_image, "rendered_text_token_ids": rendered_text_token_ids, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tuple_to_dict_w_ocr_text_image(data):
    """
    the input data is a tuple of (visual, input_ids, attention_mask, labels)
    visual is 6D input: (batch_size, Image in same chunk, Time, num_channels, height, width)
    """
    visual, input_ids, attention_mask, labels = data[0].unsqueeze(1).unsqueeze(1), data[1][0], data[1][1], data[1][2]
    rendered_text_image = data[2].unsqueeze(1).unsqueeze(1)
    rendered_text_token_ids = input_ids
    return {"visual": visual, "rendered_text_image": rendered_text_image, "input_ids": input_ids, "rendered_text_token_ids": rendered_text_token_ids, "attention_mask": attention_mask, "labels": labels}


def txt_to_dict(data):
    B = data[0][0].shape[0]
    visual, input_ids, attention_mask, labels = torch.ones(B, 1, 1, 3, 224, 224), data[0][0], data[0][1], data[0][2]
    return {"visual": visual, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def list_to_dict(data):
    """
    the input data is a tuple of (visual, input_ids, attention_mask, labels)
    visual is 6D input: (batch_size, Image in same chunk, Time, num_channels, height, width)
    """
    # Flatten the list of dictionaries
    flat_data = [item for sublist in data for item in sublist]
    return {"visual": torch.stack([d['visual'] for d in flat_data]).unsqueeze(2),
            "rendered_text_image": torch.stack([d['rendered_text_image'] for d in flat_data]).unsqueeze(2),
            "input_ids": torch.stack([d['input_ids'] for d in flat_data]).squeeze(1), 
            "attention_mask": torch.stack([d['attention_mask'] for d in flat_data]).squeeze(1), 
            "labels": torch.stack([d['labels'] for d in flat_data]).squeeze(1)}

def list_to_dict_w_text_image(data):
    """
    the input data is a tuple of (visual, input_ids, attention_mask, labels)
    visual is 6D input: (batch_size, Image in same chunk, Time, num_channels, height, width)
    """
    # Flatten the list of dictionaries
    flat_data = [item for sublist in data for item in sublist]
    return {"visual": torch.stack([d['visual'] for d in flat_data]).unsqueeze(2),
            "rendered_text_image": torch.stack([d['rendered_text_image'] for d in flat_data]).unsqueeze(2),
            "rendered_text_token_ids": torch.stack([d['rendered_text_token_ids'] for d in flat_data]).squeeze(1),
            "input_ids": torch.stack([d['input_ids'] for d in flat_data]).squeeze(1), 
            "attention_mask": torch.stack([d['attention_mask'] for d in flat_data]).squeeze(1), 
            "labels": torch.stack([d['labels'] for d in flat_data]).squeeze(1)}


def inter_video_list_to_dict(data):
    """
    the input data is a tuple of (visual, input_ids, attention_mask, labels)
    visual is 6D input: (batch_size, Image in same chunk, Time, num_channels, height, width)
    """
    # Flatten the list of dictionaries
    flat_data = [item for sublist in data for item in sublist]
    visual_token = torch.stack([d['visual'] for d in flat_data])
    input_ids = torch.stack([d['input_ids'] for d in flat_data]).squeeze(1)
    attention_mask = torch.stack([d['attention_mask'] for d in flat_data]).squeeze(1)
    labels = torch.stack([d['labels'] for d in flat_data]).squeeze(1)
    return {"visual": visual_token, "input_ids": input_ids, 
            "attention_mask": attention_mask, "labels": labels}

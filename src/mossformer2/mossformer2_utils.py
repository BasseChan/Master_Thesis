import os
import torch
import numpy as np


def load_checkpoint(checkpoint_path):
    #if use_cuda:
    #    checkpoint = torch.load(checkpoint_path)
    #else:
    checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_state_dict(model, checkpoint_dir):
    with open(os.path.join(checkpoint_dir, 'last_best_checkpoint'), 'r') as f:
        model_name = f.readline().strip()
    checkpoint_path = os.path.join(checkpoint_dir, model_name)
    checkpoint = load_checkpoint(checkpoint_path)
    if 'model' in checkpoint:
        pretrained_model = checkpoint['model']
    else:
        pretrained_model = checkpoint

    state = model.state_dict()
    for key in state.keys():
        if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
            state[key] = pretrained_model[key]
        elif key.replace('module.', '') in pretrained_model and state[key].shape == pretrained_model[key.replace('module.', '')].shape:
             state[key] = pretrained_model[key.replace('module.', '')]
        elif 'module.'+key in pretrained_model and state[key].shape == pretrained_model['module.'+key].shape:
             state[key] = pretrained_model['module.'+key]
        # elif self.print: print(f'{key} not loaded')
    model.load_state_dict(state)
    # print('=> Reload well-trained model {} for decoding.'.format(
    #         model_name))


def decode_one_audio(model, device, inputs):
    num_spks = 2
    sampling_rate = 16000
    decode_window = 1
    one_time_decode_length = 60
    out = []
    #inputs, utt_id, nsamples = data_reader[idx]
    decode_do_segement = False
    window = sampling_rate * decode_window  #decoding window length
    stride = int(window*0.75) #decoding stride if segmentation is used
    #print('inputs:{}'.format(inputs.shape))
    b,t = inputs.shape
    if t > window * one_time_decode_length: #120:
        print('The sequence is longer than {} seconds, using segmentation decoding...'.format(one_time_decode_length))
        decode_do_segement=True ##set segment decoding to true for very long sequence

    if t < window:
        inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],window-t))],1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],padding))],1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t-window)//stride * stride
            inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],padding))],1)
    #print('inputs after padding:{}'.format(inputs.shape))
    inputs = torch.from_numpy(np.float32(inputs))
    inputs = inputs.to(device)
    b,t = inputs.shape
    if decode_do_segement: # int(1.5*window) and decode_do_segement:
        outputs = np.zeros((num_spks,t))
        give_up_length=(window - stride)//2
        current_idx = 0
        while current_idx + window <= t:
            tmp_input = inputs[:,current_idx:current_idx+window]
            tmp_out_list = model(tmp_input,)
            for spk in range(num_spks):
                tmp_out_list[spk] = tmp_out_list[spk][0,:].cpu().numpy()
                if current_idx == 0:
                    outputs[spk, current_idx:current_idx+window-give_up_length] = tmp_out_list[spk][:-give_up_length]
                else:
                    outputs[spk, current_idx+give_up_length:current_idx+window-give_up_length] = tmp_out_list[spk][give_up_length:-give_up_length]
            current_idx += stride
        for spk in range(num_spks):
            out.append(outputs[spk,:])
    else:
        out_list=model(inputs)
        for spk in range(num_spks):
            out.append(out_list[spk][0,:].cpu().numpy())

    max_abs = 0
    for spk in range(num_spks):
        if max_abs < max(abs(out[spk])):
            max_abs = max(abs(out[spk]))
    for spk in range(num_spks):
        out[spk] = out[spk]/max_abs
    return out
import gradio as gr

import argparse
from config import get_config
from SpeakerNet import *
from model import built_model
from loss import built_loss
from dataloader import loadWAV

def get_cfg():
    parser = argparse.ArgumentParser(description="vox verification")
    parser.add_argument('--cfg', type=str, default="configs/ECAPA_TDNN1_step512.yaml", metavar="FILE", help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch_size', default=None, type=int, help="batch size")
    # parser.add_argument("--resume", default="train_models/ECAPA_TDNN1steplr512_cn20220526/epoch:46,EER:1.5705,MinDCF:0.1395", type=str, help="resume path")
    parser.add_argument("--resume", default="train_models/ECAPA_TDNN1steplr512_cn20220526/epoch:120,EER:0.7194,MinDCF:0.0747", type=str, help="resume path")
    parser.add_argument('--eval', dest='eval', action='store_true', default=False, help='Eval only')
    parser.add_argument('--eval_model', default=None, type=str, help="eval model path")
    parser.add_argument("--wandb", action='store_true', default=False, help='use wandb to log ')
    parser.add_argument("--note", type=str, default="", help='wandb note')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config


cfg, args = get_cfg()
device = torch.device("cpu")
model = built_model(cfg).to(device)
loss = built_loss(cfg).to(device)
model = SpeakerNet(cfg, model=model, loss=loss)

ckpt = torch.load(cfg.MODEL.RESUME, map_location="cpu")
model.load_state_dict(ckpt['model_state_dict'], strict=False)
print("checkpoint加载完毕!")

model.eval()

def SpeakerVerification(path1,path2):
    inp1 = loadWAV(path1, max_frames=300, evalmode=True)
    inp2 = loadWAV(path2, max_frames=300, evalmode=True)
    # print(inp1.shape)
    # print(inp1)
    inp1 = torch.FloatTensor(inp1)
    inp2 = torch.FloatTensor(inp2)
    # if len(inp1.shape) != 2:
    #     inp1 = torch.mean(inp1, dim=2)
    #     inp2 = torch.mean(inp2, dim=2)
    # print(inp1.shape)
    # print(inp1)
    with torch.no_grad():
        emb1 = model(inp1).detach().cpu()
        emb2 = model(inp2).detach().cpu()
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    dist = F.cosine_similarity(emb1.unsqueeze(-1),  emb2.unsqueeze(-1).transpose(0, 2)).numpy()
    # dist = F.cosine_similarity(emb1.unsqueeze(1),  emb2.unsqueeze(0), dim=-1).numpy()
    score = numpy.mean(dist)
    print(score)
    # threshold = 0.414
    if score >= 0.21:
        output = "同一个人"
    else:
        output = "不同的人"

    return output

def WrappedSpeakerVerification(path1, path2):
    output = SpeakerVerification(path1, path2)
    return {'output': output}

def launch_gradio():
    inputs = [
        gr.inputs.Audio(source="microphone", type="filepath", label="Speaker #1", optional=True),
        gr.inputs.Audio(source="microphone", type="filepath", label="Speaker #2", optional=True)
    ]
    description = (
        "This demo will compare two speech samples and determine if they are from the same speaker. "
        "Try it with your own voice!"
    )
    examples = [["example/speaker1-1.wav", "example/speaker1-2.wav"],
                ["example/speaker1-1.wav", "example/speaker2-1.wav"],
                ["example/speaker2-1.wav", "example/speaker2-2.wav"],
                ["example/speaker1-2.wav", "example/speaker2-2.wav"],
                ["example/speaker3-1.wav", "example/speaker3-2.wav"],
                ["example/speaker3-1.wav", "example/speaker4-1.wav"],
                ["example/speaker4-1.wav", "example/speaker4-2.wav"],
                ["example/speaker3-2.wav", "example/speaker4-2.wav"],
                ["example/speaker4-1.wav", "example/speaker5-2.wav"],
                ]
    iface = gr.Interface(fn=SpeakerVerification, inputs=inputs, outputs="text", examples=examples,
                        title="speaker verification based on ECAPA_TDNN", description=description)
    # iface = gr.Interface(fn=SpeakerVerification, inputs=inputs, outputs="text",
    #                     title="speaker verification based on ECAPA_TDNN", description=description)
    iface.launch()

if __name__ == '__main__':
    launch_gradio()
    # import soundfile
    # audio, sample_rate = soundfile.read("example/speaker3-1.wav")
    # print(audio, sample_rate)
    # inp1 = loadWAV("example/speaker5-1.wav", max_frames=300, evalmode=True)
    # print(inp1)
    # print(SpeakerVerification("example/speaker1-1.wav", "example/speaker1-2.wav"))
    pass

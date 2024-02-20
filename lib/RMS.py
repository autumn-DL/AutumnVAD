import torch
@torch.no_grad()
def get_rms(
        y,
        *,
        frame_length=2048,
        hop_length=512,
        pad_mode="constant",
):
    '''

    :param y: T
    :param frame_length: int
    :param hop_length: int
    :param pad_mode:
    :return: T
    '''
    # padding = (int(frame_length // 2), int(frame_length // 2))
    padding= (int((frame_length - hop_length) // 2),
                int((frame_length - hop_length+1) // 2))

    y=torch.nn.functional.pad(y, padding, pad_mode)
    y_f=y.unfold(0, frame_length, hop_length)
    y_o=((y_f**2).sum(1)/frame_length).sqrt()
    return y_o
def db2rms( threshold: float = -40.,):

    return 10 ** (threshold / 20.)
# if __name__ == "__main__":
#     # import librosa
#     sx=torch.randn(8192)
#     # R1=librosa.feature.rms(sx.numpy())
#     RMSx=get_rms(sx).numpy()
#     pass

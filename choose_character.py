import numpy as np

def character_choice(character):
    print(f'=> Choose character: {character}')

    if character.lower() == 'l36_233':
        img_postfix = '.jpg'
        n_rig=61
        data_path = '/project/qiuf/DJ01/L36/images'
        mouth_left, mouth_right, mouth_top, mouth_bottom = map(int, np.array([150, 350, 350, 450]) * 256 / 512.)
        eye_left, eye_right, eye_top, eye_bottom = map(int, np.array([106, 404, 161, 266]) * 256 / 512.)
        ckpt_img2rig = None
        ckpt_rig2img = '/project/qiuf/expr-capture/ckpt/rig2img_20240211-045412.pt'
    elif character.lower() == 'l36_234':
        img_postfix = '.jpg'
        n_rig = 61
        data_path = '/project/qiuf/DJ01/L36_234/images'
        mouth_left, mouth_right, mouth_top, mouth_bottom = map(int, np.array([178, 360, 363, 451]) * 256 / 512.)
        eye_left, eye_right, eye_top, eye_bottom = map(int, np.array([119, 415, 136, 250]) * 256 / 512.)
        ckpt_img2rig = None
        ckpt_rig2img = '/project/qiuf/expr-capture/ckpt/rig2img_20240324-184156.pt'
    elif character.lower() == 'l36_230_61':
        img_postfix = '.png'
        n_rig = 67
        data_path = '/project/qiuf/DJ01/L36_230_61/images'
        mouth_left, mouth_right, mouth_top, mouth_bottom = map(int, np.array([194, 295, 312, 367]) * 256 / 512.)
        eye_left, eye_right, eye_top, eye_bottom = map(int, np.array([151, 337, 185, 261]) * 256 / 512.)
        ckpt_img2rig = '/project/qiuf/expr-capture/ckpt/img2rig_20240522-153804.pt'
        ckpt_rig2img = '/project/qiuf/expr-capture/ckpt/rig2img_20240425-180631.pt'
    else:
        raise NotImplementedError

    mouth_crop = np.zeros((3,256,256))
    mouth_crop[:, mouth_top:mouth_bottom, mouth_left:mouth_right] = 1
    mouth_crop[:, eye_top:eye_bottom, eye_left:eye_right] = 1
    return {
        'data_path': data_path,
        'mouth_crop':mouth_crop,
        'n_rig':n_rig,
        'ckpt_img2rig': ckpt_img2rig,
        'ckpt_rig2img': ckpt_rig2img,
        'img_postfix': img_postfix
        }
    
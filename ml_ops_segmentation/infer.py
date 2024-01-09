import os
import shutil
from pathlib import Path

import cv2 as cv
import hydra
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
from omegaconf import DictConfig
from skimage.transform import resize


BASE_PATH = str(Path(__file__).parent.parent / "configs")


@hydra.main(
    config_path=BASE_PATH,
    config_name="main",
    version_base="1.2",
)
def infer(cfg: DictConfig):

    os.system(cfg.dvc.pull)
    base_workdir = Path(__file__).parent.parent
    test_path = base_workdir / cfg.data_path.test_data
    list_of_test = sorted(test_path.glob("*"))
    path_to_ans = base_workdir / cfg.data_path.result_data
    size = (cfg.resize.size, cfg.resize.size)

    if os.path.exists(path_to_ans):
        shutil.rmtree(path_to_ans)

    os.mkdir(path_to_ans)
    onnx_download_path = (
        base_workdir / cfg.data_path.onnx_save_path / cfg.data_path.name_model
    )
    ort_session = onnxruntime.InferenceSession(onnx_download_path)

    for i in range(len(list_of_test)):
        image = plt.imread(list_of_test[i] / "slice.jpg")
        image_resize = resize(
            image,
            size,
            mode=cfg.resize.mode,
            anti_aliasing=True,
        )
        image_inp = np.array([image_resize], np.float32)
        inp = np.rollaxis(image_inp, 3, 1)
        dim_back = (image.shape[1], image.shape[0])
        outputs = ort_session.run(None, {"modelInput": inp})
        mas = (outputs[0][0][0] > cfg.pred_threshold).astype(np.uint8)
        resized_back = cv.resize(mas, dim_back, interpolation=cv.INTER_AREA)
        cur_ans = str(list_of_test[i]).split("/")[-1] + cfg.data_path.name_pred_ending
        path_save = path_to_ans / cur_ans
        plt.imsave(path_save, resized_back)


if __name__ == "__main__":
    infer()

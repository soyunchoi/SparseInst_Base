# SparseInst_Base
<b> [Samsung_SparseInst_Model] </b>

* <b> Original paper:</b> https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Sparse_Instance_Activation_for_Real-Time_Instance_Segmentation_CVPR_2022_paper.pdf
* <b> Original code:</b> https://github.com/hustvl/SparseInst

## Code Updates

* [2023/05/03] Add real-time actionable code with Realsense camera in `demo.py`


# Setting
My env: RTX 3060, CUDA 11.2

<b> Setting conda env: </b>

```bash
conda create -n sparseinst python=3.8
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# install pytorch
python -m pip install detectron2==0.3 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
# install detectron2 v0.3
pip install opencv-python opencv-contrib-python scipy
pip install iopath fvcore portalocker yacs timm pyyaml==5.1 shapely
# pip install other package
git clone https://github.com/soyunchoi/SparseInst_Base.git
# SparseInst code download
cd SparseInst_Base
mkdir output
# Create directory for storing results
```
 
 # Inference
 <b> Checkpoints downlaod </b>
 
| model | backbone | input | aug | AP<sup>val</sup> |  AP  | FPS | weights |
| :---- | :------  | :---: | :-: |:--------------: | :--: | :-: | :-----: |
| [SparseInst (G-IAM)](configs/sparse_inst_r50_giam_aug.yaml) | [R-50](https://drive.google.com/file/d/1Ee6nPXlj1eewAnooYtoPtLzbRp_mDxfB/view?usp=sharing) | 608 | &#10003; | 34.2 | 34.7 | 44.6 | [model](https://drive.google.com/file/d/1MK8rO3qtA7vN9KVSBdp0VvZHCNq8-bvz/view?usp=sharing) |

<sup>&#x021A1;</sup>: measured on RTX 3090.

다운로드 이후, SparseInst_Base/weights 폴더 만들어서 안에 넣기

After downloading, create a SparseInst_Base/weights folder and place it inside.

<b> Testing </b>

* <b> Video </b>
```bash
python demo.py --config-file <CONFIG> --input <IMAGE-PATH> --output results --opts MODEL.WEIGHTS <MODEL-PATH>
# example
python demo.py --config-file configs/sparse_inst_r50_giam.yaml --video-input video.mp4 --output output --opt MODEL.WEIGHTS weights/sparse_inst_r50_giam_aug_2b7d68.pth INPUT.MIN_SIZE_TEST 512
```

* <b> Webcam </b>
```bash
python demo.py --config-file configs/sparse_inst_r50_giam.yaml --webcam --opt MODEL.WEIGHTS weights/sparse_inst_r50_giam_aug_2b7d68.pth INPUT.MIN_SIZE_TEST 512
```

# Error Handling
만약 아래와 같은 오류가 떴다면 아래의 해결방법을 통해 해결 가능

If you encounter an error like the one below, you can resolve it using the following solution.

<b> Error 1) </b>
```bash
File "/data/anaconda3/envs/sparseinst/lib/python3.8/site-packages/detectron2/utils/video_visualizer.py", line 85, in <listcomp>
    _DetectedInstance(classes[i], boxes[i], mask_rle=None, color=None, ttl=8)
TypeError: 'NoneType' object is not subscriptable
```
<b> Sol) 들여쓰기 주의 Pay attention to indentation </b> 

  Go to `"/data/anaconda3/envs/sparseinst/lib/python3.8/site-packages/detectron2/utils/video_visualizer.py"`
  and replace Line 76-87 with the following:
  ```bash
        if predictions.has("pred_masks"):
            masks = predictions.pred_masks
            # mask IOU is not yet enabled
            masks_rles = mask_util.encode(np.asarray(masks.permute(1, 2, 0), order="F"))
            assert len(masks_rles) == num_instances

            detected = [
                _DetectedInstance(classes[i], None, mask_rle=masks_rles[i], color=None, ttl=8)
                for i in range(num_instances)
            ]

        else:
            masks = None

            detected = [
                _DetectedInstance(classes[i], boxes[i], mask_rle=None, color=None, ttl=8)
                for i in range(num_instances)
            ]
  ```
  
 <b> Error 2) </b>
 ```bash
  File "/data/anaconda3/envs/sparseinst/lib/python3.8/site-packages/detectron2/utils/video_visualizer.py", line 198, in _assign_colors
    is_crowd = np.zeros((len(instances),), dtype=np.bool)
  File "/data/anaconda3/envs/sparseinst/lib/python3.8/site-packages/numpy/__init__.py", line 305, in __getattr__
    raise AttributeError(__former_attrs__[attr])
AttributeError: module 'numpy' has no attribute 'bool'.
`np.bool` was a deprecated alias for the builtin `bool`. To avoid this error in existing code, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
 ```
 <b> Sol) </b>
 Go to `"/data/anaconda3/envs/sparseinst/lib/python3.8/site-packages/detectron2/utils/video_visualizer.py", line 198`
 and edit content below, Line 198:
 ```bash
         is_crowd = np.zeros((len(instances),), dtype=np.bool_)
 ```
 

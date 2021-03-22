# ContainerNumberDist
基本能用的半成品预测代码  
# Sample
<P>
   <img src="https://github.com/kekekahuatian/ContainerNumberDist/blob/master/samples/mask.png" width="300" height="500" alt="网不好或者图没了"/>
   <img src="https://github.com/kekekahuatian/ContainerNumberDist/blob/master/samples/pred.png" width="300" height="500" alt="网不好或者图没了"/>
 </p>  
 
# Predict  

1. [predict.py](https://github.com/kekekahuatian/ContainerNumberDist/blob/master/predict.py)目前只支持单张图片
2. 修改 `--recModel识别模型路径`、`--decModel检测模型路径`、`--img_path图片路径`、`--alphabets字典路径` 来进行检测  
`python predict.py --recModel path/to/recModel --decModel path/to/decModel --img_path path/to/img_path --alphabets path/to/alphabets  
`
# Todo  
* [x] 编号检测  
* [x] 编号识别  
# Reference
https://github.com/WenmuZhou/PAN.pytorch  
https://github.com/whai362/PSENet
# 请给reference的那两个项目star！

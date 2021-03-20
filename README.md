# ContainerNumberDist
集装箱编号识别，佛系干活  
# Sample
<P>
   <img src="https://github.com/kekekahuatian/ContainerNumberDist/blob/master/samples/mask.png" width="300" height="500" alt="网不好或者图没了"/>
   <img src="https://github.com/kekekahuatian/ContainerNumberDist/blob/master/samples/pred.png" width="300" height="500" alt="网不好或者图没了"/>
 </p>  
 
# Predict  
[predict.py](https://github.com/kekekahuatian/ContainerNumberDist/blob/master/predict.py)  
1. 可以检测单张图片，也可以检测文件夹中的图片  
2. 修改 `img` or `imgFiles` 来检测  
`python predict.py --model path/to/model --imgFiles or img path/to/img --save path/to/save  
`
# Todo
<input type="checkbox" disabled=""  checked=""> 文字检测
# Reference
https://github.com/WenmuZhou/PAN.pytorch  
https://github.com/whai362/PSENet
# 请给reference那两个项目star！

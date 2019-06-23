<h1>1.基于Tensorflow的全连接神经网络：正方教务系统验证码</h1>

简介：通过post提交上来的验证码图片，通过简单图片处理裁剪，通过训练后的程序识别，并将验证码识别结果返回。

运行环境：Python3

运行必要库文件：Tensorflow、numpy、opencv-python、bottle、pillow

识别成功率：99%

识别效率：1000次/0.5s

目录结构：

  root：<br/>
&nbsp;Data            #用于存放裁剪处理后的验证码图片<br/>
        newimg          #用于存放简单处理后的验证码图片<br/>
        update          #用于将用户上传的验证码存放在此<br/>
        checkpro.exe    #傻瓜式检验此进程内存消耗过大，结束并重启进程<br/>
        code.py         #源文件<br/>
        

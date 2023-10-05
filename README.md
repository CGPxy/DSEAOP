# DSEU-net

![DSEU-net](https://github.com/CGPxy/DSEAOP/assets/52651150/d505743a-a347-4f1f-a497-4e35bf3eed38)

In this work, we developed a dee supervision U-net with SE block.


# Running Method 1

Download the weights we have already trained directly.
链接：https://pan.baidu.com/s/1kKKkrb3O8s7actVyT7ow9w 
提取码：CGPN


## step 1 pull ours image
Image ID：a258a301dfe8     
REPOSITORY：aopdsenet
TAG：latest

## step 2 uppload input image
./input/images/pelvic-2d-ultrasound/

## step 3 get segmentation result
docker run -it (your Image ID)  /bin/bash 


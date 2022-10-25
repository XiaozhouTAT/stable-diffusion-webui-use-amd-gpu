@[TOC](在linux中使用A卡进行ai模型训练)

# 吐槽
rocm都更新这么多版本了怎么还没有windows的
~~##RX580用户看过来
rocm4.0版本后就不支持RX580了，垃圾AMD
## 使用的设备配置
linux:Ubuntu20.04.1
CPU:R9-5900hx
GPU:RX6800M 12G
python:3.10.6
### 2022-10-24 23:21:50一键部署工具发布
顺序：1-8-2-3-4-5-7-6
加个源：deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main
下载链接https://www.123pan.com/s/xW39-dVMmH提取码:2333
# 安装GPU驱动
如果你已经安装成功了gpu驱动可以跳过
如果之前装过其它版本没有驱动成功的，在终端输入
`sudo amdgpu-install --uninstall`卸载驱动

访问[amd官网](https://www.amd.com/en/support/kb/release-notes/rn-amdgpu-unified-linux-22-20)下载amdgpu-install_xxxxxx.xxxxxx_all.deb

进入安装包所在的目录
接着在终端输入：`sudo apt install ./amdgpu-install_xxxxxxx-xxxxxx_all.deb`（注：amdgpu-install_xxxxxxx-xxxxxx_all.deb指的是你下载的amdgpu版本

然后`sudo apt update`再`sudo apt upgrade -y`

开始安装驱动
```shell
sudo amdgpu-install --no-dkms
sudo apt install rocm-dev
//安装完后重启
sudo reboot
```

配置环境
```shell
ls -l /dev/dri/render*
sudo usermod -a -G render $LOGNAME
sudo usermod -a -G video $LOGNAME
sudo reboot
```
**测试**
```shell
# 显示gpu性能监控
rocm-smi
#查看显卡信息的两条命令（直接在终端输入）
/opt/rocm/bin/rocminfo
/opt/rocm/opencl/bin/clinfo
#有一条报错可能是没安装好
```
## 添加path
echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64' | sudo tee -a /etc/profile.d/rocm.sh
# 安装MIopen

```shell
#安装hip
sudo apt-get install miopen-hip
#下载miopenkernels，适用与gfx1030的a卡，如果你不是可以试一下
链接：https://www.123pan.com/s/xW39-oyMmH
sudo dpkg -i miopenkernels-gfx1030-36kdb_1.1.0.50200-65_amd64.deb
```

# RDNA2架构安装pytorch
```shell
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.1.1
```
## RX580(gfx803)用户安装这个
```shell
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm3.7
```

# 运行stable-diffusion-webui
```shell
sudo apt install git
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui
#一般会提示pip版本太低，更新一下
python -m pip install --upgrade pip wheel
pip install -r requirements.txt' -i https://pypi.tuna.tsinghua.edu.cn/simple
HSA_OVERRIDE_GFX_VERSION=10.3.0 python launch.py --precision full --no-half
#HSA_OVERRIDE_GFX_VERSION可以模拟版本可以填9.0.0或者8.0.3（没试过）
//一般来讲会提示没有模型，如果有扔./models/Stable-diffusion里，本文不提供，自行百度
```
### 提示cuda错误，解决方法
torch is not able to use gpu
```python
#打开launch.py找到这句代码
commandline_args = os.environ.get('COMMANDLINE_ARGS', "")
#改成
commandline_args = os.environ.get('COMMANDLINE_ARGS', "--skip-torch-cuda-test")
```
# 疑难杂症解决
## rocm-gdb依赖libpython3.8解决
进软件和更新——其他软件——添加下面软件源

```shell
deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main
```
更新一下软件源
```powershell
sudo apt upgrade
sudo apt update
```
安装libpython3.8并重新运行amdgpu-install

```powershell
sudo apt install libpython3.8
sudo apt install rocm-dev
```
## rocm-llvm依赖python但无法安装它
找个目录进行操作
```shell
apt download rocm-llvm
ar x rocm-llvm_xxxx.xxxxx_amd64.deb
tar xf control.tar.xz
#编辑文件，如果没有vim将先安装sudo apt install vim
vim control
#找到如下一行：
Depends: python, libc6, libstdc++6|libstdc++8, libstdc++-5-dev|libstdc++-7-dev, libgcc-5-dev|libgcc-7-dev, rocm-core
#改为如下内容：
Depends: python3, libc6, libstdc++6|libstdc++8, libstdc++-5-dev|libstdc++-7-dev|libstdc++-10-dev, libgcc-5-dev|libgcc-7-dev|libgcc-10-dev, rocm-core
#重新打包
tar c postinst prerm control | xz -c > control.tar.xz
ar rcs rocm-llvm.deb debian-binary control.tar.xz data.tar.xz
#安装前先安装依赖
sudo apt install libstdc++-10-dev libgcc-10-dev rocm-core
#安装
sudo dpkg -i rocm-llvm.deb
#重新安装驱动
sudo amdgpu-install --no-dkms
```
## 运行launch.py时出现语法错误/切换python版本版本
多半是你ubuntu默认python不对应

```shell
#先查看本地安装了多少个python
ls /usr/bin/python*
#正常来讲会出现一下内容
#/usr/bin/python      /usr/bin/python3.10-config  /usr/bin/python3-futurize
#/usr/bin/python3     /usr/bin/python3.8          /usr/bin/python3-pasteurize
#/usr/bin/python3.10  /usr/bin/python3-config 
#我们要用的是python3.10的，所以
sudo rm /usr/bin/python  #删除原来的链接
sudo ln -s /usr/bin/python3.10 /usr/bin/python    #创建新的链接
python --version    #测试
```
## Can't run without a checkpoint. Find and place a .ckpt file into any of those locations. The program will exit.
你没有模型，把模型放进/models/Stable-diffusion里面吧（cpkt文件）
## 安装完驱动重启黑屏

启动的时候选择第二项(recovery模式)后，再选第一项继续进入系统，进来后卸载驱动
## 运行后下载插件超时
下载插件的速度三取决与年访问github是否流畅，很卡的话就修改launch.py吧
例

```powershell
gfpgan_package = os.environ.get('GFPGAN_PACKAGE', "git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379")
修改成
gfpgan_package = os.environ.get('GFPGAN_PACKAGE', "git+ https://ghproxy.com/https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379")
```
## GPU看戏（指GPU不工作）
用root环境运行webui吧（没试过）

```shell
su
#输入密码，如果没设置就用sudo passwd root设置密码
HSA_OVERRIDE_GFX_VERSION=10.3.0 python launch.py --precision full --no-half
#HSA_OVERRIDE_GFX_VERSION可以模拟版本可以填9.0.0或者8.0.3（没试过）
```

# 愉快玩耍
进webui目录执行以下操作

```powershell
HSA_OVERRIDE_GFX_VERSION=10.3.0 python launch.py --precision full --no-half
```
如果运行时出现什么hip错误找不到gfx1030或者其他版号的可以不用管，等待一会就可以了，后面生成就不会提示，（每次启动第一次运行都会这样）

## 显卡监控（选装）

```shell
sudo apt install radeontop
radeontop
```

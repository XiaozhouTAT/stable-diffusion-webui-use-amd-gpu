@[TOC](在linux中使用A卡进行ai模型训练)

# 吐槽
rocm都更新这么多版本了怎么还没有windows的

## 使用的设备配置
linux:Ubuntu20.04
CPU:R9-5900hx
GPU:RX6800M 12G
python:3.10.6
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
sudo amdgpu-install --usecase=dkms
amdgpu-install -y --usecase=rocm
//安装完后重启
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
配置环境
```shell
ls -l /dev/dri/render*
sudo usermod -a -G render $LOGNAME
sudo usermod -a -G video $LOGNAME
sudo reboot
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
sudo amdgpu-install
```
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
sudo amdgpu-install
```

# 安装pytorch
```shell
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.1.1
```
# 安装MIopen

```shell
//安装hip
apt-get install miopen-hip

```

# 运行stable-diffusion-webui
```shell
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
cd stable-diffusion-webui
#一般会提示pip版本太低，更新一下
python -m pip install --upgrade pip wheel
HSA_OVERRIDE_GFX_VERSION=10.3.0 python launch.py --precision full --no-half
//一般来讲会提示没有模型，如果有扔./models/Stable-diffusion里，本文不提供，自行百度
```
### 提示cuda错误，解决方法
```python
#打开launch.py找到这句代码
commandline_args = os.environ.get('COMMANDLINE_ARGS', "")
#改成
commandline_args = os.environ.get('COMMANDLINE_ARGS', "--skip-torch-cuda-test")
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

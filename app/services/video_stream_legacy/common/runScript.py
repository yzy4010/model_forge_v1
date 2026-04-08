import sys, subprocess, shutil, os
from model.videoInput import linuxServer


def ubuntu_sudorun_script(scripts, local_password: str = None, shell: bool = False):
    if local_password is not None:
        print(f"echo '{local_password}' | sudo -S {scripts}")
        subprocess.run(f"echo '{local_password}' | sudo -S {scripts}", shell=shell)
    else:
        print(scripts)
        subprocess.run(scripts, shell=shell)



def run_script(scripts, shell: bool = False):
    subprocess.run(scripts, shell=shell)


def copy_overwrite(src, dst):
    """
    复制文件或目录到目标路径，覆盖同名文件。
    
    参数:
        src (str): 源文件或目录路径
        dst (str): 目标路径
    """
    # 规范化路径，处理可能的斜杠问题
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)

    try:
        if os.path.isfile(src):
            # 处理目标路径是目录的情况
            if os.path.isdir(dst):
                dst = os.path.join(dst, os.path.basename(src))
            # 复制文件，覆盖已存在的文件
            shutil.copy2(src, dst)
            print(f"文件复制成功: {src} -> {dst}")
        else:
            # 复制目录，允许覆盖已存在的文件（需要Python 3.8+）
            # if os.path.exists(dst):
            #     # 移除目标目录下的所有内容，以实现覆盖效果
            #     shutil.rmtree(dst)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"目录复制成功: {src} -> {dst}")
    except Exception as e:
        print(f"复制过程中发生错误: {str(e)}")
        raise


def rsync_push(local_path, remote_path, remoteServer: linuxServer, local_password: str = None, shell: bool = False):
    '''
    使用rsync进行文件同步-推模式（本地推另外一台服务器）
    '''
    # https://www.cnblogs.com/nathaninchina/articles/2754109.html   -rsync+ssh+密码登录,同步文件夹
    # https://www.cnblogs.com/Jimc/p/18830333   rsync 命令核心功能与基本语法
    # local_path 若源路径末尾斜杠 / 表示同步目录内的内容，目标目录不存在时会在目标服务器自动创建
    # local_path 若源路径不带斜杠（如 /data/source），目标目录会创建以 source 为名的子目录。
    # remote_path 无所谓带不带斜杠
    scriptStr = f"sshpass -p '{remoteServer.password}' rsync -rz -e 'ssh -p {remoteServer.port}' {local_path} {remoteServer.user}@{remoteServer.ip}:{remote_path}"
    ubuntu_sudorun_script(scriptStr,None,shell)
    # echo 'glory123!@#' | sudo -S sshpass -p 'Dcom123!@#' rsync -rz -e 'ssh -p 22'  /www/logs/ dcom@192.168.1.179:/home/dcom/nacos/logs


def scp_push(local_path, remote_path, remoteServer: linuxServer, local_password: str = None, shell: bool = False):
    '''
    使用scp进行文件同步-推模式（本地推另外一台服务器）
    '''
    # https://www.cnblogs.com/nathaninchina/articles/2754109.html   -rsync+ssh+密码登录,同步文件夹
    # https://blog.csdn.net/qq_23564667/article/details/132080700   scp 命令
    # scp中local_path和remote_path无所谓/后缀（如 /data/source，或/data/source/），目标目录（就是远程目录）会创建以 source 为名的子目录。
    # ubuntu_sudorun_script(f"scp {local_path} {remote_path}",local_password,shell)
    scriptStr = f"sshpass -p '{remoteServer.password}' scp -r -P '{remoteServer.port}' {local_path} {remoteServer.user}@{remoteServer.ip}:{remote_path}"
    ubuntu_sudorun_script(scriptStr,None, shell)
    # echo 'glory123!@#' | sudo -S sshpass -p 'Dcom123!@#' scp -r -P '22'  /www/logs/ dcom@192.168.1.179:/home/dcom/nacos/logs


def rsync_pull(local_path, remote_path, remoteServer: linuxServer, local_password: str = None, shell: bool = False):
    '''
    使用rsync进行文件同步-拉模式（从另外一台服务器同步文件到本机）
    '''
    # remote_path 若源路径末尾斜杠 / 表示同步目录内的内容，目标目录不存在时会在本地服务器自动创建
    # remote_path 若源路径不带斜杠（如 /data/source），目标目录会创建以 source 为名的子目录。
    # local_path 无所谓带不带斜杠
    scriptStr = f"sshpass -p '{remoteServer.password}' rsync -rz -e 'ssh -p {remoteServer.port}' {remoteServer.user}@{remoteServer.ip}:{remote_path} {local_path}"
    ubuntu_sudorun_script(scriptStr,local_password,  shell)
    # echo 'glory123!@#' | sudo -S sshpass -p 'Dcom123!@#' rsync -rz -e 'ssh -p 22'  /www/logs/ dcom@192.168.1.179:/home/dcom/nacos/logs


def scp_pull(local_path, remote_path, remoteServer: linuxServer, local_password: str = None, shell: bool = False):
    '''
    使用scp进行文件同步-拉模式（从另外一台服务器同步文件到本机）
    '''
    # scp中local_path和remote_path无所谓/后缀（如 /data/source，或/data/source/），目标目录（就是本地目录）会创建以 source 为名的子目录。
    scriptStr = f"sshpass -p '{remoteServer.password}' scp -r -P '{remoteServer.port}' {remoteServer.user}@{remoteServer.ip}:{remote_path} {local_path}"
    ubuntu_sudorun_script( scriptStr,local_password, shell)
    # echo 'glory123!@#' | sudo -S sshpass -p 'Dcom123!@#' scp -r -P '22' dcom@192.168.1.179:/home/dcom/nacos/logs /www/logs/

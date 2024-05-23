# Deep_learning_0_2_1
In this project we are going to implement deep learning project step by step
Download dataset from ![link](https://www.kaggle.com/datasets/abtabm/multiclassimagedatasetairplanecar/download?datasetVersionNumber=2)

# 完成这些项目：
1. 图像分类
    - 简单多分类
    - 复杂多分类（数据不平衡，噪音）
2. 图像分割
3. 目标识别
4. 自然语言处理（文本分类）
    - 情绪分类
5. 时间序列预测
    - 预测类问题
6. 生成模型（GAN,AE,VAE）

6. 完成**1**的基础上再进行如下安排：
   - 分布式训练：学习如何在多个GPU或分布式环境下训练深度学习模型（训练技巧）
   - 模型优化和压缩：研究模型剪枝、量化、知识蒸馏等技术，以提高模型的推理速度和减少内存占用（训练技巧）
   - AutoML：学习自动化机器学习技术，使用工具如AutoKeras、Google AutoML等进行模型选择和超参数优化（训练技巧）
   - 云服务：熟悉AWS、Google Cloud、Azure等云平台的AI服务，学习如何在云上部署和管理模型（工程和部署）
   - 边缘计算：研究边缘设备上的深度学习应用，使用TensorFlow Lite、ONNX等框架在移动设备和物联网设备上部署模型（工程和部署）
   - 持续集成和持续部署（CI/CD）：学习如何在机器学习项目中实现CI/CD，使用工具如GitHub Actions、Jenkins等进行自动化部署和测试 （工程和部署）


# Git, DVC, 以及数据库介绍
下面是一些基本内容，并非所有的内容都会用到，因为我已经完成了其中的大部分，数据集已经是最新的了，不用额外的操作，但是你需要了解
记得每次进行开发之后都要进行 *git commit -m*, 对于团队工作来说很重要，可以清楚的知道每次自己都干了什么。此外，你需要做的就是
知道如何创建分支，然后自己的branch下面进行开发
## 数据库
我用的是 [Google Drive](https://drive.google.com/drive/folders/1NE2MCMWE6OlvFni-B71KC4zwO8vEsr7d?usp=drive_link) 作为我们远程的库，用于储存所有的 `img` 文件、`model.pth`、和其他的一些大型数据。储存在谷歌云中的三分类文件夹的 ID 地址是：`1TZ-RKDRaU4iwbZaDrIEOTe5TBvhnlCJH`。以后我们数据就都存放在里面。

## DVC (Data Version Control) 和 Git
DVC 用于模型开发和数据版本控制，它与 Git 一起使用。掌握以下命令即可：

- `dvc add <your_file>`: 添加文件到本地仓库
- `dvc commit`: 提交文件到 DVC 本地仓库
- `dvc push`: 从本地仓库推送到远程仓库
- `dvc pull`: 从远程仓库拉取到本地
- `dvc checkout <version>`: 切换版本

- `git add <file>`: 添加文件到 Git 暂存区
- `git commit -m "message"`: 提交文件到 Git 本地仓库
- `git push origin <branch>`: 推送到远程仓库
- `git status`: 查看当前仓库状态
- `git log`: 查看提交历史

### 示例 1
以下是一个使用 DVC 和 Git 创建新分支的示例（创建你自己的分支，避免直接操作主分支 `main` 以防误操作）：

1. 新建分支并自动跳转到新分支 `develop`：
    ```sh
    git checkout -b develop
    ```

2. 检查当前分支：
    ```sh
    git branch  # 确保在 develop 分支
    ```

以后你就在 `develop` 分支进行模型开发。在每次开发之前，建议你使用以下命令将 `main` 分支的最新内容拉取下来，然后合并到 `develop` 分支：
    ```sh
    git checkout develop  # 确保自己在 develop 分支
    git pull origin main  # 拉取最新的 main 分支内容
    git merge main  # 将最新的 main 分支与 develop 分支进行合并
    ```

3. 添加文件并推送：
    ```sh
    git add <your_file>
    git commit -m "commit_message"
    git push origin develop
    ```

### 示例 2
以下是一个使用 DVC 和 Git 进行开发的完整示例：

1. 初始化 DVC 项目并配置远程存储：
    ```sh
    dvc init
    dvc remote add -d myremote gdrive://1TZ-RKDRaU4iwbZaDrIEOTe5TBvhnlCJH
    # 如果是第一次使用，还需要将你的 Google Cloud 的凭证添加上
    dvc remote modify myremote gdrive_client_id <你的 Google Cloud ID>
    dvc remote modify myremote gdrive_client_secret <你的 Google Cloud Secret>
    ```

2. 添加数据文件到 DVC：
    ```sh
    dvc add data
    git add data.dvc .gitignore
    git commit -m "Add data version 1"
    ```

3. 推送到远程仓库：
    ```sh
    dvc push
    git push origin develop
    ```

4. 从远程拉取仓库：
    ```sh
    git clone git@github.com:Charly168/Deep_learning_0_2_1.git
    # 或者如果已经在仓库下就使用
    git pull origin develop
    dvc pull  # 拉取数据，一般只需要拉取一次
    ```

### 示例 3
以下是一个使用 DVC 和 Git 进行数据版本控制的完整示例：

1. 本地数据库的更新：
    如果你进行了本地数据库的更新（包括模型文件等），则使用：
    ```sh
    dvc add <your_new_data>
    dvc commit
    dvc push
    
    git add <your_new_data>.dvc
    git commit -m "data version 2"
    git push origin develop
    ```

2. 追溯到老版本的数据或模型：
    如果你觉得老版本的数据或模型更好，想要追溯回老版本：
    ```sh
    git log  # 查看历史提交，找到目标版本的 commit hash
    git checkout <commit_hash>  # 切换到目标版本
    dvc checkout  # 切换 DVC 数据到目标版本
    ```

通过这些步骤，你可以确保数据文件和版本控制在 DVC 和 Git 中都保持同步，并能够追踪和管理数据的不同版本。

# 数据集下载说明
## 先决条件1. 安装DVC: `pip install dvc`
2. 克隆此GitHub仓库: `git clone git@github.com:Charly168/Deep_learning_0_2_1.git` or 'git clone https://github.com/Charly168/Deep_learning_0_2_1.git'

## 配置DVC远程
1. 进入仓库目录: `cd Deep_learning_0_2_1`
2. 添加DVC远程:   
 ```sh    
 dvc remote add -d myremote gdrive://1TZ-RKDRaU4iwbZaDrIEOTe5TBvhnlCJH
 dvc remote modify myremote gdrive_client_id <your-client-id>    
 dvc remote modify myremote gdrive_client_secret <your-client-secret>    
 ```
Please go to [如何创建google cloud api](https://docs.zfile.vip/advanced/google-drive-api/) set the google drive API
## 认证与下载数据集
1. 在执行`dvc pull`命令前，DVC会提示进行Google账户登录。按照提示完成身份验证。
2. 执行DVC pull命令: `dvc pull`
确保你已经配置好了Google Drive的共享权限，并在需要时提供具体的文件夹ID。用户也可以使用他们自己的OAuth 2.0凭证来配置DVC远程。


## 一些有用的链接
1. [如何创建google cloud api](https://docs.zfile.vip/advanced/google-drive-api/)
2. [DVC 入门详细介绍](https://blog.51cto.com/liferecords/5132658)
3. [Git 入门详细介绍](https://zhuanlan.zhihu.com/p/369486197)
4. [DVC+google drive Introduction](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive)

# Model Tracking Serviceの利用例(ML編)
 
## 概要

Model Tracking Serviceの利用方法について、機械学習を行う基本シナリオを通しご紹介します。  
簡単なサンプルデータセットとしてMLflowの[Documentation](https://mlflow.org/docs/latest/index.html)に記載されている[sklearn_elasticnet_wine](https://github.com/mlflow/mlflow/tree/master/examples/sklearn_elasticnet_wine)を使用します。  
大規模言語モデルでファインチューニングを行うシナリオについては[LLM編](./samples-llm-finetune.md)をご確認ください。

## 事前準備

### MLflowのUIの確認

ABCIグループ単位で共有利用できるMLflow Serverが作成済みである場合に、そのUIの確認方法を説明します。  

1. [OnDemand](https://ood-portal.abci.ai/)にアクセスします。
    * ABCIアカウントでログインします。OTPの入力も必要です。 
1. `[AI Hub]` - `[MLflow Server]` を選択します。
1. 対象ServiceのURL(from outside ABCI)をクリックします。
1. BASIC認証用のユーザ名とパスワードを入力します。
1. MLflowのUIを確認します。
  
  * ※ ブラウザはChromeを推奨します。

### MLWFツールのインストール

MLWFツールにはMLflowモデルをABCIデータセットに登録するための`登録情報作成支援ツール`と、登録されたデータセットから実際に推論を行う手助けとなる`コンテナイメージ作成ツール`があります。  
MLWFツールインストール用のPythonの仮想環境(仮想環境名: mlwf)を作成し、MLWFツールをコピーしてインストールします。

```
[username@es1 ~]$ mkdir aihub
[username@es1 ~]$ cd aihub
[username@es1 aihub]$ module load python/3.11
[username@es1 aihub]$ python3 -m venv venv/mlwf
[username@es1 aihub]$ source venv/mlwf/bin/activate
(mlwf) [username@es1 aihub]$ cp -pr /apps/aihub/abci_mlwf .
(mlwf) [username@es1 aihub]$ pip install ./abci_mlwf/
(mlwf) [username@es1 aihub]$ pip install --upgrade pip
```

## 1. 学習フェーズ

### 学習環境の準備

ABCIのインタラクティブジョブを実行し、計算ノードの利用を開始します。

```
[username@es1 ~]$ qrsh -g grpname -l rt_C.small=1 -l h_rt=1:00:00
```

学習用のPythonの仮想環境としてvenvでMLWFツール用とは別の環境(仮想環境名: work)を作り、MLflowのサンプルプログラムの前提ソフトウェア`mlflow[extras]`を`pip install`します。

```
[username@g0001 ~]$ cd aihub
[username@g0001 aihub]$ module load python/3.11
[username@g0001 aihub]$ python3 -m venv venv/work
[username@g0001 aihub]$ source venv/work/bin/activate
(work) [username@g0001 aihub]$ pip install mlflow[extras]
(work) [username@g0001 aihub]$ pip install --upgrade pip
```

MLflow Tracking Serverを利用するために環境変数を設定します。  
`MLFLOW_TRACKING_URI`は、App for MLflow Serverの`URL for access from inside ABCI`に表示されている文字列をご指定ください。  
`MLFLOW_TRACKING_USERNAME`と`MLFLOW_TRACKING_PASSWORD`には、MLflow Tracking Serverのベーシック認証用の文字列をご指定ください。  
`MLFLOW_S3_ENDPOINT_URL`にはABCIクラウドストレージのURLを指定します。

環境変数の設定例)

```
(work) [username@g0001 aihub]$ export MLFLOW_TRACKING_URI="http://＜コンテナ管理サーバのIPアドレス＞:＜ポート番号＞/mlflow/"
(work) [username@g0001 aihub]$ export MLFLOW_TRACKING_USERNAME="BASIC_USERNAME"
(work) [username@g0001 aihub]$ export MLFLOW_TRACKING_PASSWORD="BASIC_PASSWORD"
(work) [username@g0001 aihub]$ export MLFLOW_S3_ENDPOINT_URL="https://s3.abci.ai"
```

### 学習処理の実行

MLflowのサンプルプログラム(mlflow)を`git clone`します。  
学習プログラム[train.py](https://github.com/mlflow/mlflow/blob/master/examples/sklearn_elasticnet_wine/train.py)を実行し、同時に学習履歴と学習済みモデルをMLflow Tracking Serverに記録します。

以下の例では、train.pyプログラムの引数に2つのパイパーパラメータalphaとl1_ratioを設定し、2度実行しています。出力された`Model name`と`version`を覚えておきます。

学習処理の実行例)

```
(work) [username@g0001 aihub]$ git clone https://github.com/mlflow/mlflow
(work) [username@g0001 aihub]$ cd mlflow/examples
(work) [username@g0001 examples]$ python sklearn_elasticnet_wine/train.py 1 1
Elasticnet model (alpha=1.000000, l1_ratio=1.000000):
  RMSE: 0.8328280500742669
  MAE: 0.672504249193216
  R2: 0.017246911564584355
Registered model 'ElasticnetWineModel' already exists. Creating a new version of this model...
2024/07/24 13:57:09 INFO mlflow.store.model_registry.abstract_store: Waiting up to 300 seconds for model version to finish creation. Model name: ElasticnetWineModel, version 32
Created version '32' of model 'ElasticnetWineModel'.

(work) [username@g0001 examples]$ python sklearn_elasticnet_wine/train.py 0.1 0.2
Elasticnet model (alpha=0.100000, l1_ratio=0.200000):
  RMSE: 0.7201489594275661
  MAE: 0.5525324524014098
  R2: 0.26518433811823017
Registered model 'ElasticnetWineModel' already exists. Creating a new version of this model...
2024/07/24 13:57:42 INFO mlflow.store.model_registry.abstract_store: Waiting up to 300 seconds for model version to finish creation. Model name: ElasticnetWineModel, version 33
Created version '33' of model 'ElasticnetWineModel'.
```

!!! note
    Jupyter Labを使用して学習を行いたい場合は、[Open OnDemand](https://ood-portal.abci.ai/)にログインし、`[Interactive Apps] - [Jupyter Notebook]`から利用が可能です。
    その場合、サンプルの[train.ipynb](https://github.com/mlflow/mlflow/blob/master/examples/sklearn_elasticnet_wine/train.ipynb)をコピーし、MLFlow Tracking Serverを利用するために環境変数を`mlflow.set_tracking_uri`関数などで同様にご指定ください。

## 2. データセット登録・公開フェーズ

### 公開対象のモデル確認

[OnDemand](https://ood-portal.abci.ai/)の`[AI Hub]` - `[MLflow Server]`から対象サービスのURLをクリックし、MLflowのUIへアクセスします。(詳細は事前準備の手順)

MLflowのUIの[Experiments]メニューから、Experiments毎やRun毎のハイパーパラメータや評価スコア(Metrics)を確認し、公開対象の学習済みモデルを決定します。
公開対象モデルの`Name`と`Version`を確認します。

### モデルの抽出

学習処理で作成されたMLflow形式のmodelに対し、登録情報作成支援ツール(`mlwf_export_model`)を使用して登録に必要なファイルを抽出します。

`mlwf_export_model` コマンドでは、`--model-registry-url`オプションで対象コンテナURL(`ポート番号`を含む)を指定します。`--model-name`オプションと`--model-version`オプションで、先ほど確認したモデル名とモデルバージョンを指定します。

モデル抽出の実行例)

```
[username@es1 ~]$ cd aihub
[username@es1 aihub]$ module load python/3.11
[username@es1 aihub]$ source venv/mlwf/bin/activate
(mlwf) [username@es1 aihub]$ export PYTHONPATH=${PYTHONPATH}:"mlwf/lib/python3.11":"abci_mlwf"
(mlwf) [username@es1 aihub]$ export MLFLOW_TRACKING_USERNAME="BASIC_USERNAME"
(mlwf) [username@es1 aihub]$ export MLFLOW_TRACKING_PASSWORD="BASIC_PASSWORD"
(mlwf) [username@es1 aihub]$ export MLFLOW_S3_ENDPOINT_URL="https://s3.abci.ai"
(mlwf) [username@es1 aihub]$ mlwf_export_model --model-registry-url="http://＜コンテナ管理サーバのIPアドレス＞:＜ポート番号＞/mlflow/" --model-name="ElasticnetWineModel" --model-version="32" --stacktrace
```

| オプション | 説明 |
|:--|:--|
| --model-registry-url | MLflow Tracking ServerのURL(`ポート番号`を含むことに注意)を指定します。 |
| --model-name | 抽出するモデルの名前を指定します。(学習処理の実行時に`Model name`に表示された名前です。) |
| --model-version | 抽出するモデルのバージョンを指定します。(学習処理の実行時に`version`に表示された数値です。) |

抽出したファイル一式は`MLWFExportModel_YYYYMMDDhhmmss`ディレクトリ配下に格納されます。  
以下のとおり確認できます。

抽出したファイルの確認例)

```
(mlwf) [username@es-a2 aihub]$ ls -go MLWFExportModel_20240724140146/
total 5
drwxr-x--- 3 4096 Jul 24 14:01 artifacts
-rw-r----- 1 3167 Jul 24 14:01 info.yaml
-rw-r----- 1 1614 Jul 24 14:01 model.tar.gz
```

| 項目 | 説明 |
|:--|:--|
| artifacts | 抽出した学習済みモデルパッケージファイル |
| info.yaml | ABCIへの登録申請用のYAML雛形 |
| model.tar.gz | artifactsディレクトリをtar.gz形式に固めたモデルパッケージファイル |

### ABCIデータセットへのモデルの登録・公開

出力された学習済みモデルパッケージ( `model.tar.gz`)をABCIクラウドストレージに配置し、登録申請用のYAML雛形`info.yaml`ファイルを編集し、[ABCI Datasets](https://datasets.abci.ai/)へ登録してください。登録方法の詳細については[ABCIユーザガイド](https://docs.abci.ai/ja/abci-datasets/)をご参照ください。

ABCIデータセットへの学習済みモデルの登録例)

```
(mlwf) [username@es1 aihub]$ module load aws-cli
(mlwf) [username@es1 aihub]$ aws --endpoint-url https://s3.abci.ai s3 mb s3://mlwf-examples/sklearn_elasticnet_wine
(mlwf) [username@es1 aihub]$ aws --endpoint-url https://s3.abci.ai s3 cp MLWFExportModel_20240724140146/model.tar.gz s3://mlwf-examples/sklearn_elasticnet_wine/
upload: MLWFExportModel_20240724140146/model.tar.gz to s3://mlwf-examples/sklearn_elasticnet_wine/model.tar.gz

(mlwf) [username@es1 aihub]$ aws --endpoint-url https://s3.abci.ai s3 ls s3://mlwf-examples/sklearn_elasticnet_wine/model.tar.gz
2024-07-24 14:04:12       1614 model.tar.gz
```

## 3. モデル利用フェーズ

### Singularityイメージファイルの作成

ABCIデータセットに登録された学習済みモデルを実行するためには、必要なパッケージがインストールされた実行環境を構築する必要があります。コンテナイメージ作成ツール `mlwf_create_image` コマンドを利用することで、第３者が開発したモデルを実行できるSingularityイメージファイルを用いて容易に実行環境を構築し、推論処理を実行することが可能です。  
`mlwf_create_image` コマンドでは、`--model-pkg-url`オプションで学習済みモデルパッケージのURLを、`--base-container-url`オプションでベースコンテナイメージを指定します。`mlwf_create_image` コマンドは、今回の例ではコンテナイメージ作成に8分ほど要します。

Singularityイメージファイルの作成例)

```
[username@es1 ~]$ qrsh -g grpname -l rt_C.small=1 -l h_rt=1:00:00

[username@g0001 ~]$ cd aihub
[username@g0001 aihub]$ module load python/3.11 singularitypro
[username@g0001 aihub]$ source venv/mlwf/bin/activate
(mlwf) [username@g0001 aihub]$ export PYTHONPATH=${PYTHONPATH}:"mlwf/lib/python3.11":"abci_mlwf"
(mlwf) [username@g0001 aihub]$ export MLFLOW_S3_ENDPOINT_URL="https://s3.abci.ai"
(mlwf) [username@g0001 aihub]$ mlwf_create_image --model-pkg-url s3://mlwf-examples/sklearn_elasticnet_wine/model.tar.gz --base-container-url docker://nvcr.io/nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

(snip)

INFO:    Creating SIF file...
INFO:    Build complete: ./MLWFCreateImage_20240724161319/container.simg
```

| オプション | 説明 |
|:--|:--|
| --model-pkg-url | 学習済みモデルパッケージのURLを指定します。ローカルファイルのパスも指定可能です。 |
| --base-container-url | ベースコンテナイメージのURLを指定します。 |

作成されたSingularityイメージと学習済みモデルパッケージは`MLWFCreateImage_YYYYMMDDhhmmss`ディレクトリ配下に格納されます。  
以下のとおり確認できます。

作成されたSingularityイメージの確認例)

```
(mlwf) [username@g0001 aihub]$ ls -goh MLWFCreateImage_20240724161319/
total 7.9G
-rwxr-x--- 1 4.8G Jul 24 16:20 container.simg
-rw-r----- 1 1.4K Jul 24 16:13 Dockerfile
drwxr-x--- 2 4.0K Jul 24 16:13 model
-rw-r----- 1 1.8K Jul 24 16:13 Singularity
```

| 項目 | 説明 |
|:--|:--|
| container.simg | Singularityイメージファイル |
| model | 学習済みモデルパッケージが可能されたディレクトリ |
| Singularity | Singularityイメージファイルを生成するために使用されたRecipeファイル |

### 公開モデルを利用した推論

計算ノードでイメージファイル`container.simg`を指定しSingularityコンテナを起動します。

Singularityコンテナの起動例)

```
(mlwf) [username@g0001 aihub]$ deactivate
[username@g0001 aihub]$ module load singularitypro
[username@g0001 aihub]$ singularity shell MLWFCreateImage_20240724161319/container.simg
```

`python --version`コマンドや`mlflow --version`コマンドで推論に必要なパッケージがインストールされた実行環境となっている事を確認します。

また`model`ディレクトリに学習済みモデル(`model.pkl`)が格納されている事も確認します。

実行環境の確認例)

```
Singularity> python --version
Python 3.11.9

Singularity> mlflow --version
mlflow, version 2.14.3

Singularity> ls -go MLWFCreateImage_20240724161319/model/
total 20
-rw-r----- 1 1359 Jul 24 14:01 MLmodel
-rw-r----- 1  232 Jul 24 14:01 conda.yaml
-rw-r----- 1  878 Jul 24 14:01 model.pkl
-rw-r----- 1  106 Jul 24 14:01 python_env.yaml
-rw-r----- 1  111 Jul 24 14:01 requirements.txt
```

実行環境と学習済みモデルの準備が整いました。

本サンプルプログラムでは、`mlflow models serve`コマンドの`-m`オプションで学習済みモデルが格納されたディレクトリを指定し、推論を行うためのMLflowのローカル RESTサーバーをデプロイします。(Singularityコンテナにより既に実行環境が用意されているため、`--env-manager=local`を指定します。)

推論環境のデプロイ例)

```
Singularity> mlflow models serve -m MLWFCreateImage_20240724161319/model -p 1234 --env-manager=local &
[1] 2792741
2024/07/24 16:28:45 INFO mlflow.models.flavor_backend_registry: Selected backend for flavor 'python_function'
2024/07/24 16:28:45 INFO mlflow.pyfunc.backend: === Running command 'exec gunicorn --timeout=60 -b 127.0.0.1:1234 -w 1 ${GUNICORN_CMD_ARGS} -- mlflow.pyfunc.scoring_server.wsgi:app'
[2024-07-24 16:28:45 +0900] [2792767] [INFO] Starting gunicorn 22.0.0
[2024-07-24 16:28:45 +0900] [2792767] [INFO] Listening at: http://127.0.0.1:1234 (2792767)
[2024-07-24 16:28:45 +0900] [2792767] [INFO] Using worker: sync
[2024-07-24 16:28:45 +0900] [2792768] [INFO] Booting worker with pid: 2792768
```

デプロイしたMLflowのローカル RESTサーバー(http://127.0.0.1:1234/) へサンプルデータを渡し、推論処理を実行できます。以下の例では2パターンのテストデータで、推論結果が出力される事を確認します。predictionsの値が、本サンプルデータセットの目的変数であるワインのqualityを推論した値になっていれば成功です。

推論処理の実行例)

```
Singularity> curl -X POST -H "Content-Type:application/json" --data '{"dataframe_split": {"columns":["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],"data":[[6.2, 0.66, 0.48, 1.2, 0.029, 29, 75, 0.98, 3.33, 0.39, 12.8]]}}' http://127.0.0.1:1234/invocations

{"predictions": [5.505222503043477]}

Singularity> curl -X POST -H "Content-Type:application/json" --data '{"dataframe_split": {"columns":["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],"data":[[6.2, 0.66, 0.48, 1.2, 0.029, 29, 25, 0.98, 3.33, 0.39, 12.8]]}}' http://127.0.0.1:1234/invocations

{"predictions": [5.702061645985878]}
```

以上

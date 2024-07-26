# AIハブ

AIハブは、ABCI上で大規模な汎用学習済みモデルの再利用を行うことができる、データ・AIモデルの共有のためのプラットフォームです。AI資源リポジトリを中心に構成されており、いくつかのサービス群やツールセットを利用できます。AI資源リポジトリとは、膨大な学習データ、大規模汎用学習済モデル、学習処理系のコンテナイメージ、それらを活用するJupyter Notebookレシピ等のAI資源を一元的に管理する場所であり、Amazon S3互換オブジェクトストレージであるABCIクラウドストレージがその役割を果たします。

AIハブは以下の３つのサービスから構成されます。

* Datasets service
* Model tracking service
* Container image service

詳細は以下のページをご確認ください。

| ページ名 | 概要 |
|:--|:--|
| [Datasets service](../abci-datasets.md) | ABCI利用者のデータセットの公開・共有を支援するカタログサービスです。 |
| [Model tracking service](./model-tracking-service.md) | MLflowをベースにモデル構築の記録・共有だけでなく、派生モデルの開発・転移学習への応用を支援するサービスとツールセットです。 |
| [Model Tracking Serviceの利用例(ML編)](./samples-ml-wine.md) | Model Tracking Serviceについて機械学習(ML)を行う利用例をご紹介します。   |
| [Model Tracking Serviceの利用例(LLM編)](./samples-llm-finetune.md) |Model Tracking Serviceについて大規模言語モデル(LLM)のファインチューニングを行う利用例をご紹介します。|
| [Container image service](../abci-singularity-endpoint.md) | Singularity Enterpriseを利用した、ABCI利用者向けのContainer image library, Remote builderサービスです。|
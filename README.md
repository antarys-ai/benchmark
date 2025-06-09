## Antarys Benchmarks

we are covering our vector database performance via benchmarking
against [dbpedia openai dataset](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M)

we also took inspiration from the [image similarity test from milvus](https://milvus.io/docs/image_similarity_search.md)
and tested it with our database

the results are available at [our website](http://antarys.ai/benchmark)

To recreate simply download the repo and run the following

```commandline
git clone https://github.com/antarys-ai/benchmark.git
cd benchmark
pip3 install -r ./requirements.txt
cd clients
python3 ./antarysdb.py
python3 ./chroma.py
```

this will generate results for individual results for each database in the `results/DATABASE` folder

This is a fairly simple test, as time goes on we will increment the complexity of the test and add more performance
metrics.

Our server uses REST API, we are opting out for a gRPC server very soon, till then this benchmark is done against all
the other vector database servers which uses gRPC.

Here are steps to run other databases locally, you will need docker, Antarys doesn't need docker setup, binaries are
available on the website!

## Qdrant

```commandline
pip install chromadb
```

For more details [please see this page](https://docs.trychroma.com/docs/overview/getting-started)

## Qdrant

```commandline
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```

For more details [please see this page](https://qdrant.tech/documentation/quickstart/)

## Milvus

```commandline
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

For more details [please see this page](https://milvus.io/docs/install_standalone-docker.md)
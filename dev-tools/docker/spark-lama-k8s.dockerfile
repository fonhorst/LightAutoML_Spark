FROM spark-lama:3.9-3.2.0

ENTRYPOINT [ "/spark/kubernetes/dockerfiles/spark/entrypoint.sh" ]

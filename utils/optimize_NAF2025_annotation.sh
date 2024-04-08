export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT

# Partition data containing one_to_many output by apet_finale
python partition_otm_nace.py $NAMESPACE/$PATH_ONE_TO_MANY_DATA $PATH_PARTITIONED_ONE_TO_MANY_DATA

# Cluster similar textual rows to decrease volume to annotate
# python textual_similarity_clustering.py ...

# Sample data by partition
# python partition_sampling.py ...
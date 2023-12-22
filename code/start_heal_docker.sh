docker run \
	-it \
	-p 8789:8789 \
	-v /mnt/e/OneDrive\ -\ Duke\ University/HEAL_ML:/workspace/HEAL_ML  \
	-v /mnt/e/Projects/HEAL_ML_ext_storage:/workspace/HEAL_ML/ext_storage  \
	--rm \
	--name heal_ml \
	--gpus all \
	--shm-size=32g \
	heal_ml \
	jupyter notebook --allow-root --ip=0.0.0.0 --port=8789

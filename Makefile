.DEFAULT_GOAL := help

download:
	@rsync -v -r -e ssh \
		--exclude out/ \
		--exclude data/ \
		--exclude tmp/ \
		--exclude __pycache__/ \
		nct01011@dt01.bsc.es:/home/nct01/nct01011/pretrained-dogs-classification/* .

upload:
	@rsync -v -r -e ssh \
		--exclude __pycache__/ \
		--exclude .git/ \
		--exclude notebooks/ \
		--exclude pictures/ \
		--exclude .idea/ \
		--exclude tmp/ \
		./ nct01011@dt01.bsc.es:/home/nct01/nct01011/pretrained-dogs-classification

queue-task:
	@ssh nct01011@mt1.bsc.es 'cd /home/nct01/nct01011/pretrained-dogs-classification && ./launchers/launch.sh train'

debug-task:
	@ssh nct01011@mt1.bsc.es 'cd /home/nct01/nct01011/pretrained-dogs-classification && ./launchers/launch.sh debug'

view-queue:
	@ssh -t nct01011@mt1.bsc.es "watch -n1 squeue"

help:
	@echo "run <make [download|upload]> to move files from/to server"

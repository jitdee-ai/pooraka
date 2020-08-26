# Pooraka project
PyTorch wrapper for image classification

[![PyPI version](https://badge.fury.io/py/pooraka.svg)](https://badge.fury.io/py/pooraka) [![Downloads](https://pepy.tech/badge/pooraka)](https://pepy.tech/project/pooraka)

## Install

The library can be installed with pip:

    pip install pooraka

## Calculating Params and FLOPs

    import pooraka as prk
    
    flops, params = prk.get_flops_params(model, (224, 224))

## CrossEntropyLabelSmooth

	import pooraka as prk
	
	CLASSES = 1000
    criterion_smooth = prk.CrossEntropyLabelSmooth(CLASSES, 0.1)
    criterion_smooth = criterion_smooth.cuda()

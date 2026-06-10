# Model Zoo

Download or place pretrained checkpoints at:

```text
checkpoints/UPSM_6stg.pth
checkpoints/UPSM_9stg.pth
```

The loader accepts checkpoints saved either as a raw state dict or as a dictionary with a top-level `model` field. It also removes a leading `module.` prefix automatically.

#Create ensemble models from base models.
ensembleEN_model = createEnsemble(BaseEfficientNet(), baseEN_weights)
ensembleNS_model = createEnsemble(BaseEfficientNet(), baseNS_weights)

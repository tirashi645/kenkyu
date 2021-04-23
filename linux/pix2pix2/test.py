    
    reconstructed_DCGAN_model = load_model(model_dir + '/DCGAN.h5')
    reconstructed_discriminator_model = load_model(model_dir + '/discriminator.h5')
    generator_model.load_weights(model_dir + '/generator_weights.h5')
    discriminator_model.load_weights(model_dir + '/discriminator_weights.h5')
    DCGAN_model.load_weights(model_dir + '/DCGAN_weights.h5')
from keras.models import load_model
import glob

model_dir = '/media/koshiba/Data/pix2pix/model'
save_dir = '/media/koshiba/Data/pix2pix/proc/output'
input_dir = '/media/koshiba/Data/pix2pix/proc/input'

file_list = glob.glob(input_dir + '/*')

DCGAN_model = load_model(model_dir + '/DCGAN.h5')
discriminator_model = load_model(model_dir + '/discriminator.h5')

print(DCGAN_model.summary())
print(discriminator_model.summary())
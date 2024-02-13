# vae
sbatch -J $jid "${STDOUT}.out" -e "${STDERR}.err" /home/it21902/latent-diffusion/scripts/job.sh --base configs/autoencoder/autoencoder_kl_64x64x3.yaml -t --gpus 0,

# ldm 
sbatch -J $jid "${STDOUT}.out" -e "${STDERR}.err" /home/it21902/latent-diffusion/scripts/job.sh --base configs/latent-diffusion/cin-ldm-vq-f8.yaml -t --gpus 0,

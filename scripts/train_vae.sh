sbatch -J $jid "${STDOUT}.out" -e "${STDERR}.err" /home/it21902/latent-diffusion/scripts/job.sh --base configs/autoencoder/autoencoder_sixray_kl_64x64x3.yaml -t --gpus 0,

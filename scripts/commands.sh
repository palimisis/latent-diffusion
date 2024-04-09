# vae
sbatch -J $jid "${STDOUT}.out" -e "${STDERR}.err" /home/it21902/latent-diffusion/scripts/job.sh --base configs/autoencoder/autoencoder_sixray_kl_64x64x3.yaml -t --gpus 0,

# ldm 
sbatch -J $jid "${STDOUT}.out" -e "${STDERR}.err" /home/it21902/latent-diffusion/scripts/job.sh --base configs/latent-diffusion/cin-ldm-vq-f8.yaml -t --gpus 0,

# sample diffusion
sbatch -J $jid "${STDOUT}.out" -e "${STDERR}.err" /home/it21902/latent-diffusion/scripts/sample_job.sh -r logs/2024-03-20T15-41-49_sixray/checkpoints -l logs/2024-03-20T15-41-49_sixray -n 1000 --batch_size 4 -n_c 1

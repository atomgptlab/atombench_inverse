configfile: "snakemake_config.yaml"
EXPS = config["experiments"]

for exp in EXPS:
    subworkflow exp:
        workdir: f"/job_runs/{exp}"
        snakefile: f"job_runs/{exp}/Snakefile"

rule all:
    input:
        expand("{exp}.final", exp=EXPS)

rule make_atomgpt_env:
    output:
        touch("job_runs/agpt_benchmark_alex/atomgpt_env.created")
        touch("job_runs/agpt_benchmark_jarvis/atomgpt_env.created")
    shell:
        """
        JOBID=$(sbatch --parsable agpt_benchmark_alex/conda_env.job)
        while squeue -j $JOBID &> do sleep 5; done
        echo done > {output}
        """

rule make_cdvae_env:
    output:
        touch("job_runs/cdvae_benchmark_alex/cdvae_env.created")
        touch("job_runs/cdvae_benchmark_jarvis/cdvae_env.created")
    shell:
        """
        JOBID=$(sbatch --parsable cdvae_benchmark_alex/conda_env.job)
        while squeue -j $JOBID &> do sleep 5; done
        echo done > {output}
        """

rule make_flowmm_env:
    output:
        touch("job_runs/flowmm_benchmark_alex/flowmm_env.created")
        touch("job_runs/flowmm_benchmark_jarvis/flowmm_env.created")
    shell:
        """
        JOBID=$(sbatch --parsable flowmm_benchmark_alex/conda_env.job)
        while squeue -j $JOBID &> do sleep 5; done
        echo done > {output}
        """

rule envs_ready:
    input:
        "job_runs/agpt_benchmark_alex/atomgpt_env.created"
        "job_runs/agpt_benchmark_jarvis/atomgpt_env.created"
        "job_runs/cdvae_benchmark_alex/cdvae_env.created"
        "job_runs/cdvae_benchmark_jarvis/cdvae_env.created"
        "job_runs/flowmm_benchmark_alex/flowmm_env.created"
        "job_runs/flowmm_benchmark_jarvis/flowmm_env.created"
    output:
        touch("job_runs/all_envs_ready.txt")
    shell:
        """
        echo all conda envs ready > {output}
        """

rule make_jarvis_data:
    input:
        "job_runs/all_envs_ready.txt"
    output:
        touch("tc_supercon/jarvis_data.created")
    shell:
        """
        cd tc_supercon
        dvc repro
        """

rule make_alex_data:
    input:
        "job_runs/all_envs_ready.txt"
    output:
        touch("alexandria/alex_data.created")
    shell:
        """
        cd alexandria
        dvc repro
        """

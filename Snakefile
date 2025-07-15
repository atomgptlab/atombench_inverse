EXPS = [
    "agpt_benchmark_alex",
    "agpt_benchmark_jarvis",
    "cdvae_benchmark_alex",
    "cdvae_benchmark_jarvis",
    "flowmm_benchmark_alex",
    "flowmm_benchmark_jarvis"
]
for exp in EXPS:
    module:
        name: exp
        snakefile: f"job_runs/{exp}/Snakefile"
    use rule * from exp

rule all:
    input:
        expand("job_runs/{exp}/{exp}.final", exp=EXPS),
        "charts.made"

rule make_atomgpt_env:
    output:
        touch("atomgpt_env.created")
    shell:
        """
        JOBID=$(sbatch --parsable job_runs/agpt_benchmark_alex/conda_env.job)
        while squeue -j $JOBID &> /dev/null; do sleep 5; done
        """

rule make_cdvae_env:
    output:
        touch("cdvae_env.created")
    shell:
        """
        JOBID=$(sbatch --parsable job_runs/cdvae_benchmark_alex/conda_env.job)
        while squeue -j $JOBID &> /dev/null; do sleep 5; done
        """

rule make_flowmm_env:
    output:
        touch("flowmm_env.created")
    shell:
        """
        JOBID=$(sbatch --parsable job_runs/flowmm_benchmark_alex/conda_env.job)
        while squeue -j $JOBID &> /dev/null; do sleep 5; done
        """

rule envs_ready:
    input:
        "atomgpt_env.created",
        "cdvae_env.created",
        "flowmm_env.created"
    output:
        touch("all_envs_ready.txt")
    shell:
        """
        echo 'all conda envs ready' > {output}
        """

rule make_jarvis_data:
    input:
        "all_envs_ready.txt"
    output:
        touch("jarvis_data.created")
    shell:
        """
        dvc -C tc_supercon repro
        """

rule make_alex_data:
    input:
        "all_envs_ready.txt"
    output:
        touch("alex_data.created")
    shell:
        """
        dvc -C alexandria repro
        """

rule make_stats_yamls:
    input:
        "flowmm_env.created",
        "jarvis_data.created",
        "alex_data.created"
    output:
        touch("flowmm_yamls.created")
    shell:
        """
        bash job_runs/flowmm_benchmark_alex/yamls.sh
        """

rule compile_results:
    input:
        expand("job_runs/{exp}/{exp}.final", exp=EXPS),
    output:
        touch("metrics.computed")
    shell:
        """
        cd job_runs/ && bash ../scripts/loop.sh
        """

rule make_bar_charts:
    input:
        "metrics.computed"
    output:
        touch("charts.made")
    shell:
        "cd job_runs/ && python ../scripts/bar_chart.py"

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
        expand("sentinels/{exp}.final", exp=EXPS),
        "sentinels/charts.made"

rule make_atomgpt_env:
    output:
        touch("sentinels/atomgpt_env.created")
    shell:
        """
        bash job_runs/agpt_benchmark_alex/conda_env.job
        """

rule make_cdvae_env:
    output:
        touch("sentinels/cdvae_env.created")
    shell:
        """
        bash job_runs/cdvae_benchmark_alex/conda_env.job
        """

rule make_flowmm_env:
    output:
        touch("sentinels/flowmm_env.created")
    shell:
        """
        bash job_runs/flowmm_benchmark_alex/conda_env.job
        """

rule envs_ready:
    input:
        "sentinels/atomgpt_env.created",
        "sentinels/cdvae_env.created",
        "sentinels/flowmm_env.created"
    output:
        touch("sentinels/all_envs_ready.txt")
    shell:
        """
        echo 'all conda envs ready' > {output}
        """

rule make_jarvis_data:
    input:
        "sentinels/all_envs_ready.txt"
    output:
        touch("sentinels/jarvis_data.created")
    shell:
        """
        dvc --cd tc_supercon repro
        """

rule make_alex_data:
    input:
        "sentinels/all_envs_ready.txt"
    output:
        touch("sentinels/alex_data.created")
    shell:
        """
        dvc --cd alexandria repro
        """

rule make_stats_yamls:
    input:
        "sentinels/flowmm_env.created",
        "sentinels/jarvis_data.created",
        "sentinels/alex_data.created"
    output:
        touch("sentinels/flowmm_yamls.created")
    shell:
        """
        bash job_runs/flowmm_benchmark_alex/yamls.sh
        """

rule compile_results:
    input:
        expand("sentinels/{exp}.final", exp=EXPS),
    output:
        touch("sentinels/metrics.computed")
    shell:
        """
        cd job_runs/ && bash ../scripts/loop.sh
        """

rule make_bar_charts:
    input:
        "sentinels/metrics.computed"
    output:
        touch("sentinels/charts.made")
    shell:
        "cd job_runs/ && python ../scripts/bar_chart.py"


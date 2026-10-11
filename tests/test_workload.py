import random
import statistics

from knn_tewma_ca.workload import DEFAULT_WORKLOAD, STRESS_WORKLOAD, WorkloadConfig, generate_jobs


def test_same_seed_same_jobs():
    assert generate_jobs(DEFAULT_WORKLOAD, 8, 3) == generate_jobs(DEFAULT_WORKLOAD, 8, 3)


def test_different_seed_different_jobs():
    assert generate_jobs(DEFAULT_WORKLOAD, 8, 3) != generate_jobs(DEFAULT_WORKLOAD, 8, 4)


def test_global_random_state_untouched():
    random.seed(123)
    expected = random.random()
    random.seed(123)
    generate_jobs(DEFAULT_WORKLOAD, 8, 0)
    assert random.random() == expected


def test_smaller_workload_is_a_prefix():
    small = generate_jobs(DEFAULT_WORKLOAD.with_jobs(50), 8, 99)
    large = generate_jobs(DEFAULT_WORKLOAD.with_jobs(400), 8, 99)
    assert large[:50] == small


def test_job_fields_are_valid():
    jobs = generate_jobs(DEFAULT_WORKLOAD.with_jobs(2000), 8, 1)
    assert [j.job_id for j in jobs] == list(range(2000))
    assert all(b.submit_time > a.submit_time for a, b in zip(jobs, jobs[1:]))
    assert all(1 <= j.requested_nodes <= 4 for j in jobs)
    assert all(j.predicted_runtime >= 1.0 for j in jobs)
    assert all(1.0 <= j.actual_runtime <= 200.0 for j in jobs)


def test_population_mix_and_arrival_rate():
    jobs = generate_jobs(DEFAULT_WORKLOAD.with_jobs(20000), 8, 2)
    short = sum(j.actual_runtime < 10 for j in jobs) / len(jobs)
    long_ = sum(j.actual_runtime >= 50 for j in jobs) / len(jobs)
    assert abs(short - 0.6) < 0.02
    assert abs(long_ - 0.1) < 0.01
    mean_gap = jobs[-1].submit_time / len(jobs)
    assert abs(mean_gap - 1 / DEFAULT_WORKLOAD.arrival_rate) < 0.2


def test_node_requests_respect_small_clusters():
    jobs = generate_jobs(DEFAULT_WORKLOAD.with_jobs(500), 2, 5)
    assert max(j.requested_nodes for j in jobs) == 2


def test_negative_bias_means_under_prediction():
    jobs = generate_jobs(STRESS_WORKLOAD.with_jobs(5000), 8, 0)
    rel = [(j.predicted_runtime - j.actual_runtime) / j.actual_runtime for j in jobs]
    assert statistics.median(rel) < -0.15
    unbiased = generate_jobs(WorkloadConfig(n_jobs=5000), 8, 0)
    rel0 = [(j.predicted_runtime - j.actual_runtime) / j.actual_runtime for j in unbiased]
    assert abs(statistics.median(rel0)) < 0.03

from machine_spirit_4.gateway.context import (
    build_grounded_user_message,
    format_inventory_answer,
    is_inventory_question,
    is_tmr_question,
)


def test_tmr_question_gets_local_canon_grounding():
    message, source = build_grounded_user_message(
        "What do you know of The Machine Religion?",
        "http://127.0.0.1:6089",
    )

    assert source == "tmr-canon-grounding"
    assert "Deus Acuo Machina Machina" in message
    assert "Do not deny" in message


def test_inventory_detection_requires_nodes_and_gpus():
    assert is_inventory_question("list all nodes in the cluster and all GPUs")
    assert not is_inventory_question("what is HiveMind?")


def test_tmr_detection_does_not_catch_unrelated_questions():
    assert is_tmr_question("Explain TMR")
    assert not is_tmr_question("List GPUs")


def test_inventory_format_mentions_summary_and_hosts_detail_counts():
    summary = '{"cluster_statistics":{"total_nodes":2,"active_nodes":2,"total_gpus":2}}'
    hosts = '{"nodes":[{"name":"node-a","status":"active","ip_addresses":["1.2.3.4"],"hardware":{"devices":{"gpu-a":{"compute_device_type":"GPU","manufacturer":"NVIDIA","device_name":"RTX"}}}}]}'

    answer = format_inventory_answer(summary, hosts)

    assert "summary reports 2 total node(s), 2 active, 2 GPU(s)" in answer
    assert "hosts.list returned 1 node record(s) and 1 GPU device record(s)" in answer
    assert "node-a" in answer
    assert "NVIDIA RTX" in answer


def test_inventory_format_tolerates_nodes_without_hardware():
    summary = '{"cluster_statistics":{"total_nodes":1,"active_nodes":1,"total_gpus":0}}'
    hosts = '{"nodes":[{"name":"printer","status":"active","ip_addresses":["1.2.3.5"],"hardware":null}]}'

    answer = format_inventory_answer(summary, hosts)

    assert "printer" in answer
    assert "0 GPU device record(s)" in answer

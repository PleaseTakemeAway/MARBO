import random
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import argparse
import os
from werewolf.agents import agent_registry
import yaml
import random
import json
from urllib.error import URLError
from urllib.request import urlopen


def model_cache_key(model_type, model_params):
    return json.dumps(
        {"model_type": model_type, "model_params": model_params},
        sort_keys=True,
        default=str,
    )


def model_identity_tags(model_type, model_params):
    tags = {str(model_type)}
    llm = str((model_params or {}).get("llm", ""))
    if llm:
        tags.add(llm)
    llm_lower = llm.lower()
    if "gpt-4o" in llm_lower or "gpt4o" in llm_lower:
        tags.add("gpt4o")
    if "gpt" in llm_lower:
        tags.add("gpt")
    if "sft" in str(model_type).lower() or "sft" in llm_lower:
        tags.add("sft_agent")
    return tags


def local_vllm_endpoint(model_type, model_params):
    if not ("sft" in str(model_type).lower() or "makto" in str(model_type).lower()):
        return None
    if "port" not in model_params:
        return None
    host = model_params.get("ip") or "localhost"
    port = model_params["port"]
    return f"http://{host}:{port}/v1/models"


def check_local_vllm_candidates(candidate_agent_models):
    missing = []
    checked = set()

    for model in candidate_agent_models:
        if float(model.get("sample_ratio", 0)) <= 0:
            continue
        model_type = model["model_type"]
        model_params = model.get("model_params", {})
        endpoint = local_vllm_endpoint(model_type, model_params)
        if endpoint is None or endpoint in checked:
            continue
        checked.add(endpoint)

        try:
            with urlopen(endpoint, timeout=2.0):
                pass
        except (OSError, URLError) as exc:
            missing.append((endpoint, model_params.get("llm", model_type), exc))

    if missing:
        details = "\n".join(
            f"  - {llm}: {endpoint} ({type(exc).__name__}: {exc})"
            for endpoint, llm, exc in missing
        )
        raise RuntimeError(
            "Configured local vLLM model endpoints are not reachable.\n"
            f"{details}\n"
            "Start the matching vLLM server(s), or remove/disable these candidates in the config."
        )


def eval(env, agent_list, roles_):
    print(agent_list)
    for agent in agent_list:
        agent.reset()
    done = False
    obs = env.reset(roles=roles_)
    while not done:
        current_act_idx = obs['current_act_idx']
        action = agent_list[current_act_idx - 1].act(obs)
        obs, reward, done, info = env.step(action)
    if done:
        if info['Werewolf'] == 1:
            return 'Werewolf win'
        elif info['Werewolf'] == -1:
            return 'Villager win'


def assign_agents(candidate_agent_models, env_config, args, assigined_roles, must_include):
    env_param = {
        "n_player": env_config["n_player"],
        "n_role": env_config["n_role"]
    }
    all_agent_models = {}
    role2agent_list = []
    selected_models = []
    sample_ratio = [a["sample_ratio"] for a in candidate_agent_models]
    candidate_tags = set()
    for model in candidate_agent_models:
        candidate_tags.update(model_identity_tags(model["model_type"], model.get("model_params", {})))
    active_must_include = [m for m in must_include if m in candidate_tags]

    while True:
        all_agent_models = {}
        role2agent_list = []
        selected_models = []
        for i, role in enumerate(assigined_roles):
            model = random.choices(candidate_agent_models, weights=sample_ratio, k=1)[0]
            model_type = model["model_type"]
            model_params = model["model_params"]
            model_key = model_cache_key(model_type, model_params)
            if model_key not in all_agent_models:
                # build model
                all_agent_models[model_key] = agent_registry.build(model_type, **model_params)
            role2agent_list.append(model_identity_tags(model_type, model_params))
            selected_models.append((model_key, model_type, model_params))
        if not active_must_include or any(m in tags for tags in role2agent_list for m in active_must_include):
            break

    agent_list = []
    role2agent_names = []
    for i, (role, selected_model) in enumerate(zip(assigined_roles, selected_models)):
        log_file = os.path.join(args.log_save_path, f"Player_{i + 1}.jsonl")
        model_key, model_type, model_params = selected_model
        type, agent_param = all_agent_models[model_key]
        agent = agent_registry.build_agent(type, i, agent_param, env_param, log_file)
        agent_list.append(agent)
        role2agent_names.append(str(model_params.get("llm", model_type)))
    return role2agent_names, agent_list


def main_cli(args):
    os.makedirs(args.log_save_path, exist_ok=True)
    parsed_yaml = yaml.safe_load(open(args.config))
    agent_config = parsed_yaml["agent_config"]
    all_candidate_agents = agent_config["all_candidates"]
    check_local_vllm_candidates(all_candidate_agents)
    env_config = parsed_yaml["env_config"]
    parent_directory = os.path.dirname(args.log_save_path)
    if not os.path.exists(os.path.join(parent_directory, "config.yaml")):
        with open(os.path.join(parent_directory, "config.yaml"), "w") as f:
            yaml.dump(parsed_yaml, f)
    env_config["log_save_path"] = args.log_save_path
    env = WerewolfTextEnvV0(**env_config)
    # assign role and models
    roles = ["Werewolf"] * env_config["n_werewolf"] + ["Villager"] * env_config["n_villager"] + \
            ["Seer"] * env_config["n_seer"] + ["Witch"] * env_config["n_witch"] + \
            ["Guard"] * env_config["n_guard"] + ["Hunter"] * env_config["n_hunter"]
    random.shuffle(roles)
    print("New rollout: ", roles)

    with open("script/all_models_us.yaml", "r") as f:
        all_models = yaml.safe_load(f)
    must_include = []
    for key, value in all_models.items():
        must_include.append(value["inenv"])
    role2agent_list, agent_list = assign_agents(all_candidate_agents, env_config, args, roles,
                                                must_include=must_include)
    print("\n\n")
    for r, a in zip(roles, role2agent_list):
        print(r, "\t", a)
    # make sure role2agent_list must have training model, or repeat

    assert len(roles) == len(role2agent_list), "The length of roles and role2agent_list must be the same"

    records = []
    for i in range(len(roles)):
        record = {
            "id": i + 1,
            "role": roles[i],
            "model": role2agent_list[i]
        }
        records.append(record)

    output_file = os.path.join(args.log_save_path, 'roles_model_assignment.json')
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(records, json_file, ensure_ascii=False, indent=4)

    print(agent_list)
    begin = time.time()
    result = eval(env, agent_list, roles)
    print(time.time() - begin, result)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config',
                           type=str, default="configs/random_models.yaml",
                           help="path to the config file of the game")
    argparser.add_argument('--log_save_path', type=str, default=None)
    argparser.add_argument('--use_vllm', action='store_true', default=False,
                           help='whether to use vllm, if set, remember add '
                                'a compatible vLLM OpenAI API server'
                                'to launch a vllm server before running the experiment.')
    args = argparser.parse_args()
    main_cli(args)

# 用于对敏感词类别判断
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

tqdm.pandas()

device = "cuda"  # the device to load the model onto
# device = "cpu" # the device to load the model onto

# Now you do not need to add "trust_remote_code=True"
model = AutoModelForCausalLM.from_pretrained(
    "/data/hub/qwen/Qwen2-72B-Instruct-GPTQ-Int4",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/data/hub/qwen/Qwen2-72B-Instruct-GPTQ-Int4")


def get_sensitive_label(row):

    word = row['word']
    if word is None:
        row['label'] = "其他"
        return row
    prompt = (
        '你是一个敏感词判别专家，敏感词的类别有：色情，政治，暴恐，谩骂，赌博五个类别，现在有一批关键词需要判断敏感词的类别。\n'
        '色情常见关键词有：骚妇，骚货，淫荡自慰器，插阴，潮吹，潮喷等\n'
        '政治常见关键词有：第一代领导，灭亡中国，反共，习大大，习近平等\n'
        '暴恐常见关键词有：炸药，枪支，枪械，炸弹，暴力，恐怖袭击，恐袭等\n'
        '谩骂常见关键词有：中国猪，台湾猪，傻逼，傻b，操你妈等\n'
        '根据列举出来各个类别的敏感词，判断输入关键词的类别，仅限于给出的五个关键词的类别，如果实在不能挂靠，可以给出：其他。例如\n',
        '输入：骚妇  输出：色情',
        '输入：习大大  输出：政治',
        '输入：恐怖袭击  输出：暴恐',
        '输入：测试  输出：其他',
        '输入：{}')
    # prompt.format(word)
    messages = [
        {"role": "system", "content": "你是一个网络警察&敏感词判别专家，专门抓取网路上出现的违规违禁词语。"},
        {"role": "user", "content": prompt.format(word)}
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        # Directly use generate() and tokenizer.decode() to get the output.
        # Use `max_new_tokens` to control the maximum output length.
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # 色情，政治，暴恐，谩骂，赌博
        print(response)
        if response:
            row['label'] = response
            return row
            # return
        else:
            row['label'] = "其他"
            return row
    except Exception as e:
        print(e)
        row['label'] = "其他"
        return row


word_list = pd.read_table('暂未分类.txt', sep='\t', header=None, names=['word'])


# word_list['label'] = word_list['word'].progress_apply(get_sensitive_label)
#
# word_list.to_csv('result.csv',index=None)
# def increase_income_by_percent(row, percent_increase=10):
#     """
#     函数用于增加收入百分比。
#     """
#     row['Income'] = row['Income'] * (1 + percent_increase / 100)
#     return row


# 输出文件名
output_file = 'result.csv'
#
# # 检查文件是否存在，不存在则先写入表头
# if not pd.read_csv(output_file, nrows=0).empty:
#     word_list.iloc[0:0].to_csv(output_file, index=False)  # 写入空的DataFrame以创建文件并包含表头

# 遍历DataFrame，处理每一行并即时保存
for index, row in tqdm(word_list.iterrows(), total=len(word_list), desc="Processing rows"):
    processed_row = get_sensitive_label(row)
    processed_df = pd.DataFrame(processed_row).transpose()  # 转置以便保存单行
    processed_df.to_csv(output_file, mode='a', header=False, index=False)  # 追加到文件，不写入表头

print("处理并保存完成。")

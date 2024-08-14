from openai import OpenAI
import json

client = OpenAI(
    api_key='sk-TZ3LYEnQ7Uqr1Ch9Kugiei3lfdp4gFaHmJskpOl_eFT3BlbkFJolffCEGvKNmVkzCJcgL1eRB8aZb3o2nN-AAoUVWzIA'
)


def generate_summarized_sentences(video_id, timestamps, sentences, fps, num_frames, num_variants=5):
    # Create the prompt for GPT-4
    prompt = (
        f"Generate {num_variants} unique, clear, and concise single-sentence descriptions summarizing the key action or event "
        f"in a video. The description should be direct and focused, avoiding any mention of the kitchen or introductory phrases like "
        f"'In this video' or 'This video is about'. The description should capture the essence of the overall scene succinctly, "
        f"and should not include numbers or any other characters at the beginning. "
        f"Here's the sequence of actions from the video:\n\n"
    )

    for sentence in sentences:
        prompt += f"- {sentence}\n"

    prompt += (
        f"\nThe description should provide a direct and clear summary of the main action or event in the video, "
        f"without unnecessary detail or length. Return {num_variants} distinct single-sentence descriptions, "
        f"separated by '###', without any numbers or additional characters at the beginning of each sentence."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0.7
        )

        # Extract the generated summaries from the response
        summarized_sentences = response.choices[0].message.content.split("###")

        return [
            {
                "video_id": video_id,
                "summarized_sentence": sentence.strip(),
                "fps": fps,
                "num_frames": num_frames
            }
            for sentence in summarized_sentences
        ]

    except Exception as e:
        print(f"Error generating summarized sentences for video {video_id}: {e}")
        return []


def process_video_data(input_json_file, output_json_file):
    with open(input_json_file, 'r') as infile:
        video_data = json.load(infile)

    records_count = len(video_data.items())
    counter = 1
    output_data = []
    for video_id, details in video_data.items():
        print(f"Processing {counter}/{records_count}")
        counter += 1
        timestamps = details['timestamps']
        sentences = details['sentences']
        fps = details['fps']
        num_frames = details['num_frames']

        # Generate summarized sentences using the OpenAI API
        summarized_data = generate_summarized_sentences(video_id, timestamps, sentences, fps, num_frames)
        output_data.extend(summarized_data)

    # Write the output to a new JSON file
    with open(output_json_file, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)
        print(f"Output written to {output_json_file}")


if __name__ == "__main__":
    input_json_file = "train.json"
    output_json_file = "new_annotations_1/train.json"

    process_video_data(input_json_file, output_json_file)

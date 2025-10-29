# Invoice Extractor Tool

## What is this?
A Python CLI that monitors a folder for new invoice files and automatically extracts key information from them.

When you drop an invoice photo into a watched directory, the tool uses a chain with 2 local language models

1. [LFM2-VL-3B](https://huggingface.co/LiquidAI/LFM2-VL-3B) extracts a raw textual description from an invoice picture.

2. [LFM2-1.2B-Extract](https://huggingface.co/LiquidAI/LFM2-1.2B-Extract) tranforms the raw textual description into a structured record. This record is appended to a CSV file.

This a practical example of building agentic workflows that run entirely on your local machine: no API keys, no cloud costs, no private data shared with third-parties.

![](./media/chain_diagram.gif)

## Features

- Automatic invoice processing using [Liquid Nanos](https://huggingface.co/collections/LiquidAI/liquid-nanos).
- Structured outputs using Pydantic models for type-safe data extraction
- Real-time directory monitoring with file system events
- Two-stage extraction pipeline (image-to-text → text-to-structured data)


## Project Structure

```
invoice-parser/
├── pyproject.toml              # Python project configuration
├── uv.lock                     # Dependency lock file
├── Makefile                    # Build automation
├── README.md                   # This file
├── invoices/                   # Sample invoice images
│   ├── Sample-electric-Bill-2023.jpg
│   ├── british_gas.png
│   └── water_australia.png
├── media/                      # Documentation assets
│   └── chain_diagram.gif
└── src/                        # Source code
    └── invoice_parser/
        ├── __init__.py         # Package initialization
        ├── main.py             # CLI entry point
        ├── invoice_processor.py # AI processing pipeline
        ├── invoice_file_handler.py # File monitoring & CSV export
        └── py.typed            # Type checking marker
```

## Code Overview

### Key Components

**InvoiceProcessor**: Manages AI model pipeline
```python
class InvoiceProcessor:
    def __init__(self, extractor_model: str, image_process_model: str):
        self.extractor_model = extractor_model
        self.image_process_model = image_process_model
    
    def process(self, image_path: str) -> InvoiceData | None:
        invoice_text = self.image2text(image_path)
        return self.text2json(invoice_text)
```

**InvoiceFileHandler**: Monitors directory for new files
```python
class InvoiceFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if self._is_image_file(event.src_path):
            self.process_invoice(event.src_path)
    
    def process_invoice(self, image_path: str):
        bill_data = self.processor.process(image_path)
        self.append_to_csv(bill_data)
```

**Pydantic Models**: Type-safe data structures
```python
class InvoiceData(BaseModel):
    utility: str
    amount: float
    currency: str
```

## Environment setup

You will need

- [Ollama](https://ollama.com/) to serve the Language Models locally.
- [uv](https://docs.astral.sh/uv/) to manage Python dependencies and run the application efficiently without creating virtual environments manually.

### Install Ollama

<details>
<summary>Click to see installation instructions for your platform</summary>

**macOS:**
```bash
# Download and install from the website
# Visit: https://ollama.ai/download

# Or use Homebrew
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download the installer from [https://ollama.ai/download](https://ollama.ai/download)

</details>


### Install UV

<details>
<summary>Click to see installation instructions for your platform</summary>

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

</details>


## How to run it?

For example:
```sh
uv run python src/invoice_parser/main.py \
    --dir invoices/ \
    --image-model hf.co/LiquidAI/LFM2-VL-3B-GGUF:F16 \
    --extractor-model hf.co/LiquidAI/LFM2-1.2B-Extract-GGUF:F16 \
    --process-existing
```

## Further improvements

Here is a list of features I challenge you to implement:

- [ ] Add a system prompt in the VLM to identify and discard images that do not correspond to bills.
- [ ] Add an on_delete handler to keep in sync the list of invoices in the directory and the `bills.csv` file.


## Wanna learn more about building prod-ready local agentic workflows?

<a href="https://discord.gg/DFU3WQeaYD"><img src="https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white" alt="Join Discord"></a></a>


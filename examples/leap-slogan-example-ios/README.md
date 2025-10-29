#  A Slogan Generator iOS app

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/iOS/LeapSloganExample)

## Table of Contents

- [Introduction](#introduction)
- [This is what you will learn](#this-is-what-you-will-learn)
- [Prerequisites](#prerequisites)
- [Part 1: Understanding the Architecture](#part-1-understanding-the-architecture)
- [Part 2: Project Setup](#part-2-project-setup)
  - [Step 1: Create a New Xcode Project](#step-1-create-a-new-xcode-project)
  - [Step 2: Add LeapSDK via Swift Package Manager](#step-2-add-leapsdk-via-swift-package-manager)
  - [Step 3: Download a Model Bundle](#step-3-download-a-model-bundle)
- [Part 3: Building the ViewModel](#part-3-building-the-viewmodel)
  - [Step 3.1: Create the Basic Structure](#step-31-create-the-basic-structure)
  - [Step 3.2: Implement Model Loading](#step-32-implement-model-loading)
  - [Step 3.3: Implement Slogan Generation](#step-33-implement-slogan-generation)
- [Part 4: Building the User Interface](#part-4-building-the-user-interface)
- [Part 5: Understanding the Flow](#part-5-understanding-the-flow)
- [Part 6: Advanced Features to Explore](#part-6-advanced-features-to-explore)
  - [6.1: Add Generation Options](#61-add-generation-options)
  - [6.2: Implement Conversation History](#62-implement-conversation-history)
  - [6.3: Add System Prompts](#63-add-system-prompts)
  - [6.4: Implement Stop Functionality](#64-implement-stop-functionality)
- [Part 7: Performance Optimization Tips](#part-7-performance-optimization-tips)
- [Part 8: Troubleshooting Common Issues](#part-8-troubleshooting-common-issues)
- [Next Steps](#next-steps)

## Introduction

Welcome to this hands-on tutorial where you'll learn how to build a real iOS app that generates creative slogans using local AI models, with no internet connection required.

In this tutorial, we'll walk through the **LeapSloganExample**, a simple SwiftUI application that demonstrates the core concepts of on-device AI inference using Liquid AI's LeapSDK.

![IMAGE]()

## This is what you will learn

By the end of this guide, you'll understand:

- How to integrate the LeapSDK into your iOS project
- How to load and run AI models locally on an iPhone or iPad
- How to implement real-time streaming text generation

Let's start!


## Prerequisites

Before we begin, make sure you have:

- **Xcode 15.0+** with Swift 5.9 or later
- **iOS 15.0+** deployment target
- A **physical iOS device** (iPhone or iPad) for best performance
  - *The iOS Simulator works but will be significantly slower*
- Basic familiarity with **SwiftUI** and Swift's async/await syntax


## Part 1: Understanding the Architecture

Before we write code, let's understand what we're building. The LeapSlogan app has a clean, three-layer architecture:

```
┌─────────────────────────────────┐
│      SwiftUI View Layer         │ ← User Interface
│  (ContentView, UI Components)   │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│     ViewModel Layer             │ ← Business Logic
│  (SloganViewModel, @Observable) │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│      LeapSDK Layer              │ ← AI Inference
│  (ModelRunner, Conversation)    │
└─────────────────────────────────┘
```

**Key components we'll implement:**

1. **ModelRunner**: Manages the loaded AI model in memory
2. **Conversation**: Handles the chat context and generation
3. **Streaming API**: Receives AI output token-by-token in real-time
4. **SwiftUI View**: Displays the UI and handles user interaction


## Part 2: Project Setup

### Step 1: Create a New Xcode Project

1. Open Xcode and create a new iOS App
2. Choose **SwiftUI** for the interface
3. Set minimum deployment target to **iOS 15.0**

### Step 2: Add LeapSDK via Swift Package Manager

LeapSDK is distributed as a Swift Package, making integration straightforward:

1. In Xcode, go to **File → Add Package Dependencies**
2. Enter the repository URL:
   ```
   https://github.com/Liquid4All/leap-ios.git
   ```
3. Select the latest version (0.6.0 or newer)
4. Add **both** products to your target:
   - ✅ `LeapSDK`
   - ✅ `LeapSDKTypes`

> **Important**: Starting with version 0.5.0, you must add both `LeapSDK` and `LeapSDKTypes` for proper runtime linking.

### Step 3: Download a Model Bundle

Now we need an AI model. LeapSDK uses **model bundles** - packaged files containing the model and its configuration:

1. Visit the [Leap Model Library](https://leap.liquid.ai/models)
2. For this tutorial, download a small model like **LFM2-350M** (great for mobile, ~500MB)
3. Download the `.bundle` file for your chosen model
4. Drag the `.bundle` file into your Xcode project
5. ✅ Make sure "Add to target" is checked

Your project structure should now look like:
```
YourApp/
├── YourApp.swift
├── ContentView.swift
├── Models/
│   └── LFM2-350M-8da4w_output_8da8w-seq_4096.bundle  ← Your model
└── Assets.xcassets
```

## Part 3: Building the ViewModel

The ViewModel is the heart of our app. It manages the model lifecycle and handles generation. Let's build it step by step.

### Step 3.1: Create the Basic Structure

Create a new Swift file called `SloganViewModel.swift`:

```swift
import Foundation
import SwiftUI
import LeapSDK
import Observation

@Observable
class SloganViewModel {
    // MARK: - Published State
    var isModelLoading = true
    var isGenerating = false
    var generatedSlogan = ""
    var errorMessage: String?
    
    // MARK: - Private Properties
    private var modelRunner: ModelRunner?
    private var conversation: Conversation?
    
    // MARK: - Initialization
    init() {
        // Model will be loaded when view appears
    }
}
```

**What's happening here?**
- `@Observable` is Swift's new observation macro (iOS 17+, but works great on iOS 15 with backports)
- We track four pieces of UI state: loading, generating, the slogan text, and any errors
- `ModelRunner` and `Conversation` are private—these are our LeapSDK objects

### Step 3.2: Implement Model Loading

Add the model loading function:

```swift
// MARK: - Model Management
@MainActor
func setupModel() async {
    isModelLoading = true
    errorMessage = nil
    
    do {
        // 1. Get the model bundle URL from app bundle
        guard let modelURL = Bundle.main.url(
            forResource: "qwen-0.6b",  // Change to match your bundle name
            withExtension: "bundle"
        ) else {
            errorMessage = "Model bundle not found in app bundle"
            isModelLoading = false
            return
        }
        
        // 2. Load the model using LeapSDK
        print("Loading model from: \(modelURL.path)")
        modelRunner = try await Leap.load(url: modelURL)
        
        // 3. Create an initial conversation
        conversation = Conversation(
            modelRunner: modelRunner!,
            history: []
        )
        
        isModelLoading = false
        print("Model loaded successfully!")
        
    } catch {
        errorMessage = "Failed to load model: \(error.localizedDescription)"
        isModelLoading = false
        print("Error loading model: \(error)")
    }
}
```

**Understanding the code:**

1. **Bundle lookup**: We find the model bundle in our app's resources
2. **Async loading**: `Leap.load()` is async because loading models takes time (1-5 seconds)
3. **Conversation creation**: Every generation needs a `Conversation` object that tracks history
4. **Error handling**: We catch and display any loading failures

> **💡 Pro Tip**: Model loading is the slowest part. In production apps, show a nice loading screen!

### Step 3.3: Implement Slogan Generation

Now for the exciting part—generating slogans! Add this function:

```swift
// MARK: - Generation
@MainActor
func generateSlogan(for businessType: String) async {
    // Guard against invalid states
    guard let conversation = conversation,
          !isGenerating else { return }
    
    isGenerating = true
    generatedSlogan = ""  // Clear previous slogan
    errorMessage = nil
    
    // 1. Create the prompt
    let prompt = """
    Create a catchy, memorable slogan for a \(businessType) business. \
    Make it creative, concise, and impactful. \
    Return only the slogan, nothing else.
    """
    
    // 2. Create a chat message
    let userMessage = ChatMessage(
        role: .user,
        content: [.text(prompt)]
    )
    
    // 3. Generate response with streaming
    let stream = conversation.generateResponse(message: userMessage)
    
    // 4. Process the stream
    do {
        for await response in stream {
            switch response {
            case .chunk(let text):
                // Append each text chunk as it arrives
                generatedSlogan += text
                
            case .reasoningChunk(let reasoning):
                // Some models output reasoning - we can log it
                print("Reasoning: \(reasoning)")
                
            case .complete(let usage, let completeInfo):
                // Generation finished!
                print("✅ Generation complete!")
                print("Tokens used: \(usage.totalTokens)")
                print("Speed: \(completeInfo.stats?.tokenPerSecond ?? 0) tokens/sec")
                isGenerating = false
            }
        }
    } catch {
        errorMessage = "Generation failed: \(error.localizedDescription)"
        isGenerating = false
    }
}
```

**Breaking down the streaming API:**

The `generateResponse()` method returns an **AsyncStream** that emits three types of events:

1. **`.chunk(text)`**: Each piece of generated text arrives here
   - This is what makes the UI feel responsive!
   - Text appears word-by-word, just like ChatGPT
   
2. **`.reasoningChunk(reasoning)`**: Some models show their "thinking"
   - Advanced feature for models that explain their reasoning
   
3. **`.complete(usage, info)`**: The final event when generation finishes
   - Contains token usage statistics
   - Includes performance metrics (tokens/second)


## Part 4: Building the User Interface

Now let's create a beautiful, interactive UI! Create or modify `ContentView.swift`:

```swift
import SwiftUI

struct ContentView: View {
    @State private var viewModel = SloganViewModel()
    @State private var businessType = ""
    
    var body: some View {
        NavigationStack {
            ZStack {
                // Background gradient
                LinearGradient(
                    colors: [.blue.opacity(0.1), .purple.opacity(0.1)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
                
                VStack(spacing: 24) {
                    if viewModel.isModelLoading {
                        modelLoadingView
                    } else {
                        mainContentView
                    }
                }
                .padding()
            }
            .navigationTitle("AI Slogan Generator")
            .navigationBarTitleDisplayMode(.large)
        }
        .task {
            // Load model when view appears
            await viewModel.setupModel()
        }
    }
    
    // MARK: - Subviews
    
    private var modelLoadingView: some View {
        VStack(spacing: 20) {
            ProgressView()
                .scaleEffect(1.5)
            Text("Loading AI Model...")
                .font(.headline)
            Text("This may take a few seconds")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
    
    private var mainContentView: some View {
        VStack(spacing: 24) {
            // Error message if any
            if let error = viewModel.errorMessage {
                errorBanner(error)
            }
            
            // Instructions
            instructionsCard
            
            // Input field
            businessTypeInput
            
            // Generate button
            generateButton
            
            // Generated slogan display
            if !viewModel.generatedSlogan.isEmpty {
                sloganResultCard
            }
            
            Spacer()
        }
    }
    
    private var instructionsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("How it works", systemImage: "lightbulb.fill")
                .font(.headline)
                .foregroundColor(.blue)
            
            Text("Enter a business type and I'll generate a creative slogan using AI—completely on your device!")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.blue.opacity(0.1))
        .cornerRadius(12)
    }
    
    private var businessTypeInput: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Business Type")
                .font(.subheadline)
                .fontWeight(.semibold)
            
            TextField("e.g., coffee shop, tech startup, bakery", text: $businessType)
                .textFieldStyle(.roundedBorder)
                .autocapitalization(.none)
                .disabled(viewModel.isGenerating)
        }
    }
    
    private var generateButton: some View {
        Button(action: {
            Task {
                await viewModel.generateSlogan(for: businessType)
            }
        }) {
            HStack {
                if viewModel.isGenerating {
                    ProgressView()
                        .tint(.white)
                } else {
                    Image(systemName: "sparkles")
                }
                
                Text(viewModel.isGenerating ? "Generating..." : "Generate Slogan")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(
                businessType.isEmpty || viewModel.isGenerating 
                    ? Color.gray 
                    : Color.blue
            )
            .foregroundColor(.white)
            .cornerRadius(12)
        }
        .disabled(businessType.isEmpty || viewModel.isGenerating)
    }
    
    private var sloganResultCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Your Slogan", systemImage: "quote.bubble.fill")
                    .font(.headline)
                    .foregroundColor(.purple)
                
                Spacer()
                
                // Copy button
                Button(action: {
                    UIPasteboard.general.string = viewModel.generatedSlogan
                }) {
                    Image(systemName: "doc.on.doc")
                        .foregroundColor(.blue)
                }
            }
            
            Text(viewModel.generatedSlogan)
                .font(.title3)
                .fontWeight(.medium)
                .foregroundColor(.primary)
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.purple.opacity(0.1))
                .cornerRadius(8)
        }
        .padding()
        .background(Color.white)
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 5, y: 2)
    }
    
    private func errorBanner(_ message: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
            Text(message)
                .font(.caption)
            Spacer()
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .foregroundColor(.red)
        .cornerRadius(8)
    }
}

#Preview {
    ContentView()
}
```

**UI Design Highlights:**

1. **Progressive disclosure**: Loading screen → Main interface
2. **Clear visual feedback**: Loading states, disabled states, animations
3. **Helpful instructions**: Users understand what to do immediately
4. **Polished details**: Gradient background, shadows, rounded corners
5. **Copy functionality**: Users can easily copy the generated slogan


## Part 5: Understanding the Flow

Let's trace what happens when a user generates a slogan:

```
1. User enters "coffee shop" and taps Generate
   ↓
2. UI disables input and shows "Generating..."
   ↓
3. ViewModel creates prompt with business type
   ↓
4. ChatMessage is sent to Conversation
   ↓
5. LeapSDK starts model inference
   ↓
6. Tokens stream back one-by-one
   ├─ "Wake" → UI updates
   ├─ " up" → UI updates
   ├─ " to" → UI updates
   ├─ " flavor" → UI updates
   └─ "!" → UI updates
   ↓
7. .complete event fires
   ↓
8. UI re-enables input, shows final slogan
```

**The magic of streaming:**
- Each word appears immediately
- Users see progress in real-time
- Feels fast and responsive
- No waiting for complete generation


## Part 6: Advanced Features to Explore

Want to take your app further? Try these enhancements:

### 6.1: Add Generation Options

Control the creativity of outputs:

```swift
let options = GenerationOptions(
    temperature: 0.8,  // Higher = more creative (0.0-2.0)
    topP: 0.9,         // Nucleus sampling
    maxTokens: 50      // Limit response length
)

let stream = conversation.generateResponse(
    message: userMessage,
    options: options
)
```

### 6.2: Implement Conversation History

Build a multi-turn conversation:

```swift
// Keep track of history
private var conversationHistory: [ChatMessage] = []

func generateWithHistory(for input: String) async {
    let userMessage = ChatMessage(role: .user, content: [.text(input)])
    conversationHistory.append(userMessage)
    
    // Create conversation with history
    let conversation = Conversation(
        modelRunner: modelRunner!,
        history: conversationHistory
    )
    
    // Generate...
    // Then add assistant response to history
}
```

### 6.3: Add System Prompts

Set the model's behavior:

```swift
let systemMessage = ChatMessage(
    role: .system,
    content: [.text("You are a creative marketing expert specializing in memorable, catchy slogans.")]
)

let conversation = Conversation(
    modelRunner: modelRunner!,
    history: [systemMessage]
)
```

### 6.4: Implement Stop Functionality

Let users cancel generation:

```swift
private var generationTask: Task<Void, Never>?

func generateSlogan(for businessType: String) async {
    generationTask = Task {
        // ... generation code ...
    }
}

func stopGeneration() {
    generationTask?.cancel()
    isGenerating = false
}
```


## Part 7: Performance Optimization Tips

### Model Selection

Choose the right model for your needs:

| Model Size | Memory | Speed | Quality | Best For |
|------------|--------|-------|---------|----------|
| 350M-500M | ~500MB | Fast | Good | Quick tasks |
| 1B-2B | ~1-2GB | Medium | Better | Balanced |
| 3B+ | ~3GB+ | Slower | Best | Quality-critical |

### Loading Optimization

```swift
// Load model once at app launch, not per-view
@MainActor
class AppState: ObservableObject {
    static let shared = AppState()
    var modelRunner: ModelRunner?
    
    func loadModelOnce() async {
        guard modelRunner == nil else { return }
        modelRunner = try? await Leap.load(url: modelURL)
    }
}
```

### Memory Management

```swift
// Unload model when done
deinit {
    // ModelRunner is automatically cleaned up
    // But you can explicitly nil it
    modelRunner = nil
}
```

## Part 8: Troubleshooting Common Issues

### Issue 1: "Model bundle not found"

**Solution**: 
- Check that `.bundle` file is in Xcode project
- Verify "Target Membership" is checked
- Ensure bundle name in code matches actual filename

### Issue 2: "Failed to load model"

**Solution**:
- Test on a physical device (Simulator is unreliable)
- Ensure iOS version is 15.0+
- Check device has enough free storage (~2-3x model size)
- Try a smaller model first

### Issue 3: Slow generation speed

**Solution**:
- Use a physical device (10-100x faster than Simulator)
- Choose a smaller model (350M-1B)
- Lower `maxTokens` in GenerationOptions
- Reduce temperature for faster but less creative output

### Issue 4: App crashes on launch

**Solution**:
- Ensure both `LeapSDK` and `LeapSDKTypes` are added
- Check frameworks are set to "Embed & Sign"
- Clean build folder (Cmd+Shift+K)
- Restart Xcode


## Next Steps

Congratulations! 🎉 You've built a fully functional on-device AI app. Here are some ideas to expand your skills:

### Immediate Next Projects

1. **LeapChat**: Build a full chat interface with history
   - Check out the [LeapChatExample](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/iOS/LeapChatExample)
   
2. **Add Structured Output**: Use `@Generatable` macros
   - Generate JSON data structures
   - Validate output format at compile-time

3. **Implement Function Calling**: Let AI call your functions
   - Weather lookup, calculations, database queries
   - See [Function Calling Guide](https://leap.liquid.ai/docs/edge-sdk/ios/function-calling)

### Learning Resources

- **Official Documentation**: [leap.liquid.ai/docs](https://leap.liquid.ai/docs/edge-sdk/ios/ios-quick-start-guide)
- **Model Library**: [leap.liquid.ai/models](https://leap.liquid.ai/models)
- **Example Apps**: [github.com/Liquid4All/LeapSDK-Examples](https://github.com/Liquid4All/LeapSDK-Examples)
- **Discord Community**: Join for support and discussions


## Need Help?

- 📚 Read the [iOS Quick Start Guide](https://leap.liquid.ai/docs/edge-sdk/ios/ios-quick-start-guide)
- 💬 Join the [Discord Community](https://discord.gg/liquid-ai)
- 🐛 Report issues on [GitHub](https://github.com/Liquid4All/leap-ios/issues)
- 📧 Contact: support@liquid.ai

---

*Tutorial created for LeapSDK v0.6.0 | Last updated: October 2025*
# 🔥 RoboCup Thermal C — Cheat Sheet

## 1. Thermal Conductivity — `k`

**কোনো material-এর মধ্যে heat কত সহজে ছড়ায়।**

```text
High k                    Low k

🔥 ─────────→             🔥 ──→
heat দ্রুত ছড়ায়           heat ধীরে ছড়ায়
```

- Metal → সাধারণত high
- Plastic → সাধারণত low

### তোমাদের target-এ

3D-printed material-এর `k` heat কত দ্রুত target-এর মধ্যে ছড়াবে সেটা প্রভাবিত করে।

**মনে রাখো:**  
> `k ↑` → heat সহজে ছড়ায়।

---

# 2. Conduction

**Direct contact-এর মাধ্যমে heat transfer।**

```text
HOT HAND WARMER
████████████
      ↓
      ↓ conduction
      ↓
3D PRINT
░░░░░░░░░░░░
```

Heat সাধারণত:

\[
Hot \rightarrow Cold
\]

দিকে যায়।

### তোমাদের target-এ

Hand warmer → 3D printed target → surface temperature pattern।

---

# 3. Temperature Gradient — `∇T`

এক জায়গা থেকে আরেক জায়গায় temperature কত দ্রুত পরিবর্তিত হচ্ছে।

```text
40°C → 36°C → 32°C → 28°C → 25°C
```

এটাই temperature distribution-এর spatial change।

### কেন দরকার?

Thermal camera যে contrast দেখবে, সেটা এই temperature differences-এর ওপর নির্ভর করে।

---

# 4. IR Radiation

Thermal camera **heat touch করে মাপে না**।

Object infrared radiation emit করে।

```text
HOT TARGET
     ↓
IR radiation
     ↓
Thermal lens
     ↓
Detector
     ↓
Image
```

তাই পুরো chain:

> **Temperature → IR radiation → detector signal → thermal image**

---

# 5. Emissivity — `ε`

Surface কত efficiently infrared radiation emit করে।

Range:

\[
0 \leq \epsilon \leq 1
\]

সহজ example:

```text
Same actual temperature

Matte plastic       Shiny metal
     ↓                   ↓
different IR emission
     ↓                   ↓
different thermal appearance
```

### তোমাদের জন্য

3D printed plastic-এর surface এবং surrounding material-এর emissivity thermal image interpretation-এ প্রভাব ফেলতে পারে।

**তাই thermal camera-তে “যে color দেখছি = exact temperature” ধরে নেওয়া যাবে না।**

---

# 6. Spatial Resolution ⭐

এটা তোমাদের **সবচেয়ে গুরুত্বপূর্ণ concept**।

মানে:

> **Camera কত ছোট spatial detail আলাদা করে দেখতে পারে।**

ধরো actual target:

```text
C   GAP   C
```

High resolution:

```text
C   GAP   C
```

Low resolution:

```text
█████████
```

Inner C ছোট হতে হতে একসময় camera সেটাকে আলাদা feature হিসেবে resolve করতে পারবে না।

### RoboCup connection

**Thermal Image Acuity = এই resolving capability পরীক্ষা করার মূল উদ্দেশ্য।**

---

# 7. IFOV ⭐

**IFOV = Instantaneous Field of View**

সহজভাবে:

> **একটা detector pixel কত ছোট angular area দেখছে।**

Approximation:

\[
IFOV \approx \frac{p}{f}
\]

যেখানে:

- `p` = pixel pitch
- `f` = focal length

Target-এর distance `D` হলে:

\[
\text{IFOV footprint} \approx D \times IFOV
\]

### Example

যদি:

```text
IFOV = 2 mrad
Distance = 2 m
```

তাহলে:

\[
2 \times 0.002 = 0.004m
\]

অর্থাৎ:

### **≈ 4 mm/pixel**

এখন যদি C-এর critical gap 4 mm হয়, সেটা extremely challenging হবে।

---

# 8. FOV

**Field of View = camera একসাথে কত বড় angular area দেখতে পারে।**

```text
          Camera
            ▼
           / \
          /   \
         /     \
        /       \
       Target scene
```

Wide FOV:

```text
\             /
 \           /
  \_________/
```

Narrow FOV:

```text
   \       /
    \_____/
```

### গুরুত্বপূর্ণ

একই sensor resolution হলেও lens/FOV বদলালে target কত pixels occupy করবে বদলে যাবে।

---

# 9. Working Distance ⭐

Camera থেকে target-এর distance।

```text
Camera ───────────── Target
           D
```

Distance বাড়লে target image-এ ছোট হয়ে যায়।

ধরো:

```text
1 m → 2 mm/pixel
2 m → 4 mm/pixel
4 m → 8 mm/pixel
```

তাহলে:

**Distance ↑ → spatial detail ↓**

এজন্য RoboCup-এর জন্য শুধু camera test করলেই হবে না—**কোন distance-এ test হচ্ছে সেটাও standardize করতে হবে।**

---

# 10. NETD ⭐

**NETD = Noise Equivalent Temperature Difference**

এটা thermal sensitivity-এর গুরুত্বপূর্ণ specification।

সহজভাবে:

> Camera কত ছোট temperature difference distinguish করতে পারে।

উদাহরণ:

```text
Camera A → 100 mK
Camera B → 40 mK
```

সবকিছু equal হলে B সাধারণত ছোট thermal differences detect করতে বেশি capable।

### কিন্তু মনে রাখবে

NETD এবং spatial resolution এক জিনিস না।

```text
NETD
 ↓
Temperature difference detect

Spatial resolution
 ↓
Spatial detail distinguish
```

তোমাদের C target-এর জন্য **দুটোই দরকার।**

---

# 11. Pixel Pitch

FPA-এর adjacent detector elements-এর spacing।

ধরো:

```text
Pixel pitch = 12 µm
```

Conceptually:

```text
●────12µm────●────12µm────●
```

Pixel pitch + focal length → IFOV-তে প্রভাব ফেলে।

তাই camera specification-এ শুধু:

> `640 × 512`

দেখে camera ভালো/খারাপ বলা যাবে না।

---

# 12. Gap Width ⭐

Landolt C-এর সবচেয়ে গুরুত্বপূর্ণ feature।

```text
       █████
     ██     ██
    ██
    ██
    ██
     ██     ██
       █████
```

এখানে C-এর opening হলো **gap**।

### Gap ছোট হলে:

```text
Large gap → সহজ
Small gap → কঠিন
```

Camera-কে gap এবং surrounding heated region আলাদা করতে হবে।

**এই gap-এর physical size camera acuity-এর সঙ্গে সরাসরি সম্পর্কিত।**

---

# 13. Feature Size ⭐

Camera-কে যে **সবচেয়ে ছোট physical detail** resolve করতে হচ্ছে।

ধরো:

```text
Outer C diameter = 40 mm
Inner C diameter = 20 mm
Gap = 5 mm
```

এখানে critical feature হতে পারে:

**5 mm gap**

যদি camera target distance-এ:

```text
5 mm ≈ 1 pixel
```

তাহলে খুব কঠিন।

যদি:

```text
5 mm ≈ 5 pixels
```

তাহলে অনেক বেশি information পাওয়া যাবে।

### তাই:

> **Feature size + mm/pixel = practical difficulty বোঝার গুরুত্বপূর্ণ relationship।**

---

# 14. Concentricity ⭐

দুই C-এর center একই জায়গায়।

```text
        Outer C
      ┌─────────┐
     /           \
    |    Inner    |
    |      C      |
     \           /
      └─────────┘
```

Mathematically:

```text
Outer center ≈ Inner center
```

যদি:

```text
Outer center = (320,240)
Inner center = (322,241)
```

তাহলে প্রায় concentric।

কিন্তু:

```text
Outer center = (320,240)
Inner center = (350,260)
```

তাহলে properly concentric না।

---

# 🎯 এখন Target-টা পুরোপুরি বুঝো

ধরো তোমাদের target:

```text
                 THERMAL CAMERA
                       ↓
                       ↓ IR
          ┌─────────────────────┐
          │                     │
          │      OUTER C        │
          │       ┌─────┐       │
          │       │INNER│       │
          │       │  C  │       │
          │       └─────┘       │
          │                     │
          └─────────────────────┘
                       ↑
                3D PRINTED TARGET
                       ↑
                 HOT HAND WARMER
```

এখন ১৪টা concept একসাথে কাজ করছে:

```text
Hand warmer
     │
     │ heat
     ▼
Conduction
     │
     ▼
Thermal conductivity
     │
     ▼
Temperature distribution
     │
     ▼
Temperature gradient
     │
     ▼
IR radiation
     │
     ▼
Emissivity
     │
     ▼
Thermal optics
     │
     ├── Focal length
     ├── Focus
     └── FOV
     │
     ▼
FPA
     │
     ├── Pixel pitch
     └── Resolution
     │
     ▼
IFOV
     │
     ▼
mm/pixel @ working distance
     │
     ▼
Can camera resolve:
     │
     ├── Outer C ?
     └── Inner C ?
             ↓
       2 LEVELS DEEP
             ↓
            PASS
```

# 🧠 সবচেয়ে গুরুত্বপূর্ণ ৫টা relationship

তুমি যদি এখন সবকিছু মুখস্থ করতে না চাও, এই ৫টা আগে মাথায় ঢোকাও:

### ① Heat

**Hot → Cold**

### ② Thermal camera

**IR radiation → detector → thermal image**

### ③ Spatial resolution

**Smaller feature → harder to resolve**

### ④ Distance

**Distance ↑ → mm/pixel ↑ → fine detail harder to see**

### ⑤ Target

**Outer C + Inner C + gap → acuity test**

---

## একটা জিনিস বিশেষভাবে মনে রেখো

**NETD ভালো হলেই camera এই task ভালো করবে না।**

ধরো camera খুব sensitive:

> 0.02°C difference detect করতে পারে।

কিন্তু optics এত খারাপ যে inner C-এর 5 mm gap resolve করতে পারে না।

তাহলে:

**Thermal sensitivity = excellent**  
**Spatial acuity = poor**

এবং তোমাদের task-এ সেটা সমস্যা।

উল্টোভাবেও হতে পারে—spatial resolution ভালো, কিন্তু thermal contrast এত কম যে C দেখা যাচ্ছে না।

তাই তোমাদের task-এর আসল equation conceptually:

> **Thermal contrast + Spatial resolution + Optics + Working distance → Observable C detail**

এটাই তোমাদের project-এর core engineering problem।
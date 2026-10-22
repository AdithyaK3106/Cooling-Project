# Plug & Play Landing Page Package

This folder contains the complete, self-contained **THERVO** video landing page module. Anyone can copy this folder into their React project to get the exact same landing page.

---

## 📁 Package Contents
- `LandingPage.tsx` — Main landing page component with cybernetic glassmorphism styling, ambient glow rings, video background, and seamless navigation router hook.
- `README.md` — Integration guide.

---

## 🚀 Quick Setup (3 Steps)

### Step 1: Copy Video File
Copy your landing page video (`use_this.mp4`) into your project's `public/` directory:
```text
your-project/
└── public/
    └── use_this.mp4
```

### Step 2: Ensure Dependencies
Make sure `lucide-react` and `react-router-dom` are installed in your project:
```bash
npm install lucide-react react-router-dom
```
*(Tailwind CSS is used for styling).*

---

### Step 3: Add to Router

Import and render `<LandingPage />` at your root route (`/`):

```tsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { LandingPage } from './landing_page_package/LandingPage';

export function App() {
  return (
    <Router>
      <Routes>
        {/* Landing Page on root */}
        <Route path="/" element={<LandingPage enterTargetRoute="/dashboard" />} />
        
        {/* Your Main Project Routes */}
        <Route path="/dashboard" element={<YourProjectDashboard />} />
      </Routes>
    </Router>
  );
}
```

---

## ⚙️ Component Props (Optional)

| Prop | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `videoSrc` | `string` | `'/use_this.mp4'` | Path to video relative to `public/` folder |
| `enterTargetRoute` | `string` | `'/dual-node'` | Route path to navigate to when ENTER is clicked |
| `brandName` | `string` | `'THERVO'` | Brand header text displayed at the top |

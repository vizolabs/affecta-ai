import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: {
    default: "Human Facial Emotion Detection",
    template: "%s · Human Facial Emotion Detection",
  },
  description:
    "A ranking-based explainable AI model for human facial emotion detection. Fully local: webcam frames never leave your machine.",
  applicationName: "Human Facial Emotion Detection",
  openGraph: {
    title: "Human Facial Emotion Detection",
    description:
      "A ranking-based explainable AI model for human facial emotion detection — fully local.",
    type: "website",
    siteName: "Human Facial Emotion Detection",
  },
};

// Apply the persisted theme before paint so a light-theme user doesn't get
// a hydration mismatch / dark flash (SSR can't know localStorage).
const themeInit = `try{var t=localStorage.getItem("hfd-theme");if(t==="light"){document.documentElement.classList.remove("dark")}else if(t==="dark"){document.documentElement.classList.add("dark")}}catch(e){}`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}

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

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}

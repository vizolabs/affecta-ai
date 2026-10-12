import Header from "../../components/layout/Header";
import Protected from "../../components/auth/Protected";
import Toaster from "../../components/common/Toaster";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <Protected>
      <div className="flex min-h-screen flex-col">
        <Header />
        <main className="flex-1 p-4">{children}</main>
        <footer className="border-t border-border px-4 py-3 text-center text-[11px] text-muted">
          Human Facial Emotion Detection · 100% local inference · frames never
          leave this device · not a diagnostic tool · AU branch untrained
          (w<sub>AU</sub> = 0) · intensity is a proxy measure
        </footer>
        <Toaster />
      </div>
    </Protected>
  );
}

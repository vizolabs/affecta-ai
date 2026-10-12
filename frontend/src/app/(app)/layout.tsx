import Header from "../../components/layout/Header";
import Protected from "../../components/auth/Protected";
import Toaster from "../../components/common/Toaster";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <Protected>
      <div className="flex min-h-screen flex-col">
        <Header />
        <main className="flex-1 p-4">{children}</main>
        <Toaster />
      </div>
    </Protected>
  );
}

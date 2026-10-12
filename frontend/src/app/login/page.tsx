import { redirect } from "next/navigation";

/** The sign-in portal lives at the root page (`/`). */
export default function LoginPage() {
  redirect("/");
}

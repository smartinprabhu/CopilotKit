
import type {Metadata} from 'next';
import {Geist, Geist_Mono} from 'next/font/google';
import '../styles/globals.css'; // Updated path to globals.css
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'Capacity Insights',
  description: 'Capacity Planning Dashboard',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) { // Added children prop type
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(true); // Assume initial state
  const [isSidebarFloating, setIsSidebarFloating] = useState(false); // Assume initial state

  // You'll need to pass down the sidebar state (expanded/collapsed and floating)
  // from a parent component that manages the sidebar.
  // This example assumes the state is managed elsewhere and passed down.
  // For a functional implementation, you'd likely use Context API or similar.


  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}>
        {children}
        <Toaster />
      </body>
    </html>
  );
}
